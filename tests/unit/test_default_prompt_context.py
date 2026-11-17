"""Unit tests for the retrieved context in the default system prompt.

The default prompt lists every exposed entity as CSV. What the context
providers retrieve for a request (relevant entities with details, the
whole-house summary, memories) used to be available only to custom prompts
through ``{{ entity_context }}``; with the default prompt it was dropped.
"""

from unittest.mock import MagicMock, patch

import jinja2
import pytest

from custom_components.home_agent.const import (
    CONF_CONTEXT_MODE,
    CONF_DIRECT_ENTITIES,
    CONF_EMIT_EVENTS,
    CONF_HISTORY_ENABLED,
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_LLM_MODEL,
    CONF_PROMPT_CUSTOM_ADDITIONS,
    CONF_PROMPT_USE_DEFAULT,
    CONTEXT_MODE_DIRECT,
    CONTEXT_MODE_VECTOR_DB,
    DEFAULT_SYSTEM_PROMPT,
    MEMORY_CONTEXT_HEADER,
    VECTOR_DB_FALLBACK_MARKER,
)

MEMORIES = f"{MEMORY_CONTEXT_HEADER}\n\n- [Fact] The user prefers the bedroom at 68 degrees\n\n"
ENTITIES = '{"entities": [{"entity_id": "light.office", "state": "on"}], "count": 1}'


@pytest.fixture
def mock_hass():
    """Create a mock Home Assistant instance."""
    hass = MagicMock()
    hass.states.async_all = MagicMock(return_value=[])
    hass.data = {}
    hass.config.location_name = "Test Home"
    return hass


def _agent(mock_hass, **config):
    from custom_components.home_agent.agent import HomeAgent
    from custom_components.home_agent.conversation_session import ConversationSessionManager

    base = {
        CONF_LLM_BASE_URL: "http://localhost:11434/v1",
        CONF_LLM_API_KEY: "test-key",
        CONF_LLM_MODEL: "test-model",
        CONF_CONTEXT_MODE: CONTEXT_MODE_VECTOR_DB,
        CONF_HISTORY_ENABLED: False,
        CONF_EMIT_EVENTS: False,
    }
    base.update(config)
    with patch("custom_components.home_agent.agent.core.async_should_expose", return_value=False):
        return HomeAgent(mock_hass, base, ConversationSessionManager(mock_hass))


class TestDefaultPromptContext:
    """Which part of the retrieved context the default prompt receives."""

    def test_vector_db_mode_gets_everything(self, mock_hass):
        """Relevant entities, whole-house summary and memories all go in."""
        agent = _agent(mock_hass)
        context = f"{ENTITIES}\n{MEMORIES}"
        assert agent._default_prompt_context(context) == context

    @pytest.mark.parametrize("context", ["", "No relevant context found.", "   "])
    def test_nothing_retrieved_adds_nothing(self, mock_hass, context):
        """An empty result does not add an empty section."""
        assert _agent(mock_hass)._default_prompt_context(context) == ""

    def test_direct_mode_all_entities_gets_memories_only(self, mock_hass):
        """With no entity list, direct context repeats the CSV, so only memories are added."""
        agent = _agent(mock_hass, **{CONF_CONTEXT_MODE: CONTEXT_MODE_DIRECT})
        result = agent._default_prompt_context(f"{ENTITIES}\n{MEMORIES}")
        assert result == MEMORIES.strip()
        assert "light.office" not in result

    def test_direct_mode_all_entities_without_memories_adds_nothing(self, mock_hass):
        """No memories and a context that repeats the CSV: nothing to add."""
        agent = _agent(mock_hass, **{CONF_CONTEXT_MODE: CONTEXT_MODE_DIRECT})
        assert agent._default_prompt_context(ENTITIES) == ""

    def test_direct_mode_with_entity_list_gets_everything(self, mock_hass):
        """A configured entity list is the user's chosen detail, so it is included."""
        agent = _agent(
            mock_hass,
            **{CONF_CONTEXT_MODE: CONTEXT_MODE_DIRECT, CONF_DIRECT_ENTITIES: "light.office"},
        )
        context = f"{ENTITIES}\n{MEMORIES}"
        assert agent._default_prompt_context(context) == context

    def test_vector_db_fallback_gets_memories_only(self, mock_hass):
        """When the vector DB is down the fallback is every entity, which the CSV already has."""
        agent = _agent(mock_hass)
        context = f"{VECTOR_DB_FALLBACK_MARKER}\n{ENTITIES}\n{MEMORIES}"
        assert agent._default_prompt_context(context) == MEMORIES.strip()

    def test_build_system_prompt_passes_both_variables(self, mock_hass):
        """The default prompt gets relevant_context; entity_context stays for custom prompts."""
        agent = _agent(mock_hass, **{CONF_CONTEXT_MODE: CONTEXT_MODE_DIRECT})
        context = f"{ENTITIES}\n{MEMORIES}"
        with patch.object(agent, "_render_template", return_value="rendered") as render:
            agent._build_system_prompt(entity_context=context)

        template_str, variables = render.call_args.args
        assert template_str == DEFAULT_SYSTEM_PROMPT
        assert variables["entity_context"] == context
        assert variables["relevant_context"] == MEMORIES.strip()

    def test_custom_prompt_is_unchanged(self, mock_hass):
        """A fully custom prompt still receives the whole entity_context."""
        agent = _agent(
            mock_hass,
            **{
                CONF_CONTEXT_MODE: CONTEXT_MODE_DIRECT,
                CONF_PROMPT_USE_DEFAULT: False,
                CONF_PROMPT_CUSTOM_ADDITIONS: "Context: {{ entity_context }}",
            },
        )
        with patch.object(agent, "_render_template", return_value="rendered") as render:
            agent._build_system_prompt(entity_context=ENTITIES)

        template_str, variables = render.call_args.args
        assert template_str == "Context: {{ entity_context }}"
        assert variables["entity_context"] == ENTITIES


class TestDefaultPromptTemplate:
    """The default prompt template itself, rendered with plain Jinja."""

    @staticmethod
    def _render(**variables):
        env = jinja2.Environment()
        env.globals.update(
            area_name=lambda *_: "Office", area_id=lambda *_: "office", now=lambda: "now"
        )
        base = {"exposed_entities": [], "current_device_id": None, "relevant_context": ""}
        base.update(variables)
        return env.from_string(DEFAULT_SYSTEM_PROMPT).render(**base)

    def test_context_section_is_rendered_before_the_closing_line(self):
        """The retrieved context sits after the CSV and before the final instruction."""
        prompt = self._render(relevant_context=MEMORIES.strip())
        assert "The user prefers the bedroom at 68 degrees" in prompt
        assert prompt.index("```csv") < prompt.index("The user prefers the bedroom at 68 degrees")
        assert prompt.index("The user prefers the bedroom at 68 degrees") < prompt.index(
            "Now respond to the user's request:"
        )
        assert prompt.rstrip().endswith("Now respond to the user's request:")

    def test_no_section_when_there_is_no_context(self):
        """Without retrieved context the prompt is the same as before."""
        prompt = self._render(relevant_context="")
        assert "Retrieved For This Request" not in prompt
        assert prompt.rstrip().endswith("Now respond to the user's request:")

    def test_context_is_inserted_verbatim(self):
        """JSON context is not re-rendered or escaped."""
        prompt = self._render(relevant_context=ENTITIES)
        assert ENTITIES in prompt
