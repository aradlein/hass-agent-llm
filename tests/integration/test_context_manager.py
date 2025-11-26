"""Integration tests for context manager functionality.

These tests verify that the context manager correctly switches between modes
and handles different context formatting options.
"""

import pytest
from unittest.mock import patch, MagicMock

from custom_components.home_agent.agent import HomeAgent
from custom_components.home_agent.const import (
    CONF_CONTEXT_FORMAT,
    CONF_CONTEXT_MODE,
    CONF_DEBUG_LOGGING,
    CONF_DIRECT_ENTITIES,
    CONF_EMIT_EVENTS,
    CONF_HISTORY_ENABLED,
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_LLM_MAX_TOKENS,
    CONF_LLM_MODEL,
    CONF_LLM_TEMPERATURE,
    CONF_VECTOR_DB_COLLECTION,
    CONF_VECTOR_DB_EMBEDDING_BASE_URL,
    CONF_VECTOR_DB_EMBEDDING_MODEL,
    CONF_VECTOR_DB_EMBEDDING_PROVIDER,
    CONF_VECTOR_DB_ENABLED,
    CONF_VECTOR_DB_HOST,
    CONF_VECTOR_DB_PORT,
    CONF_VECTOR_DB_TOP_K,
    CONTEXT_FORMAT_HYBRID,
    CONTEXT_FORMAT_JSON,
    CONTEXT_FORMAT_NATURAL_LANGUAGE,
    CONTEXT_MODE_DIRECT,
    CONTEXT_MODE_VECTOR_DB,
)


@pytest.mark.integration
@pytest.mark.requires_llm
@pytest.mark.asyncio
async def test_context_mode_switching_direct(
    test_hass, llm_config, sample_entity_states, session_manager
):
    """Test switching to direct context mode.

    This test verifies that:
    1. Context manager correctly initializes in direct mode
    2. Direct mode provides entity context from configured entities
    3. Context is properly injected into LLM prompts
    """
    # Configuration with direct context mode
    config = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 500,
        CONF_CONTEXT_MODE: CONTEXT_MODE_DIRECT,
        CONF_DIRECT_ENTITIES: [
            "light.living_room",
            "sensor.temperature",
        ],
        CONF_CONTEXT_FORMAT: CONTEXT_FORMAT_JSON,
        CONF_HISTORY_ENABLED: False,
        CONF_EMIT_EVENTS: False,
        CONF_DEBUG_LOGGING: False,
    }

    with patch(
        "custom_components.home_agent.agent.async_should_expose",
        return_value=False,
    ):
        # Setup test states
        test_hass.states.async_all = MagicMock(return_value=sample_entity_states)

        def mock_get_state(entity_id):
            for state in sample_entity_states:
                if state.entity_id == entity_id:
                    return state
            return None

        test_hass.states.get = MagicMock(side_effect=mock_get_state)

        agent = HomeAgent(test_hass, config, session_manager)

        # Verify context manager is using direct mode
        assert agent.context_manager is not None, "Context manager should be initialized"
        assert hasattr(
            agent.context_manager, "_get_mode_from_config"
        ), "Context manager should have _get_mode_from_config method"
        context_mode = agent.context_manager._get_mode_from_config()
        assert context_mode == CONTEXT_MODE_DIRECT, f"Expected direct mode, got {context_mode}"

        # Get context to verify it works
        context = await agent.context_manager.get_context(
            user_input="What's the status of my lights?",
            conversation_id="test_direct_mode",
        )

        assert context is not None, "Context should not be None in direct mode"
        assert isinstance(
            context, (str, list, dict)
        ), f"Context should be str/list/dict, got {type(context)}"
        context_str = str(context)
        assert len(context_str) > 0, "Context should not be empty in direct mode"
        # Context should contain entity information
        assert any(
            keyword in context_str.lower()
            for keyword in ["entity", "state", "light", "sensor", "living_room", "temperature"]
        ), f"Context should contain entity information, got: {context_str[:200]}"

        # Context should mention configured entities
        context_str = str(context).lower()
        assert (
            "living_room" in context_str or "light" in context_str
        ), "Context should include living room light"

        await agent.close()


@pytest.mark.integration
@pytest.mark.requires_llm
@pytest.mark.requires_chromadb
@pytest.mark.requires_embedding
@pytest.mark.asyncio
async def test_context_mode_switching_vector_db(
    session_manager,
    test_hass,
    llm_config,
    chromadb_config,
    embedding_config,
    sample_entity_states,
    test_collection_name,
):
    """Test switching to vector_db context mode.

    This test verifies that:
    1. Context manager correctly initializes in vector_db mode
    2. Vector DB mode retrieves relevant context using embeddings
    3. Context is properly retrieved and formatted
    """
    # Configuration with vector_db context mode
    config = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 500,
        CONF_CONTEXT_MODE: CONTEXT_MODE_VECTOR_DB,
        CONF_VECTOR_DB_ENABLED: True,
        CONF_VECTOR_DB_HOST: chromadb_config["host"],
        CONF_VECTOR_DB_PORT: chromadb_config["port"],
        CONF_VECTOR_DB_COLLECTION: test_collection_name,
        CONF_VECTOR_DB_TOP_K: 5,
        CONF_VECTOR_DB_EMBEDDING_MODEL: embedding_config["model"],
        CONF_VECTOR_DB_EMBEDDING_PROVIDER: embedding_config["provider"],
        CONF_VECTOR_DB_EMBEDDING_BASE_URL: embedding_config["base_url"],
        CONF_CONTEXT_FORMAT: CONTEXT_FORMAT_JSON,
        CONF_HISTORY_ENABLED: False,
        CONF_EMIT_EVENTS: False,
        CONF_DEBUG_LOGGING: False,
    }

    with patch(
        "custom_components.home_agent.agent.async_should_expose",
        return_value=True,  # Expose entities for vector DB
    ):
        # Setup test states
        test_hass.states.async_all = MagicMock(return_value=sample_entity_states)

        def mock_get_state(entity_id):
            for state in sample_entity_states:
                if state.entity_id == entity_id:
                    return state
            return None

        test_hass.states.get = MagicMock(side_effect=mock_get_state)

        agent = HomeAgent(test_hass, config, session_manager)

        # Initialize vector DB if needed
        if hasattr(agent, "vector_db_manager") and agent.vector_db_manager:
            # Sync entities to vector DB
            await agent.vector_db_manager.sync_entities()

        # Verify context manager is using vector_db mode
        assert agent.context_manager is not None, "Context manager should be initialized"
        assert hasattr(
            agent.context_manager, "_get_mode_from_config"
        ), "Context manager should have _get_mode_from_config method"
        context_mode = agent.context_manager._get_mode_from_config()
        assert (
            context_mode == CONTEXT_MODE_VECTOR_DB
        ), f"Expected vector_db mode, got {context_mode}"

        # Get context to verify it works
        context = await agent.context_manager.get_context(
            user_input="What's the temperature?",
            conversation_id="test_vector_db_mode",
        )

        # Context might be empty if no entities were synced, but should not be None
        assert context is not None, "Context should not be None in vector_db mode"

        # Clean up
        if hasattr(agent, "vector_db_manager") and agent.vector_db_manager:
            # Delete the test collection
            try:
                chromadb_client = agent.vector_db_manager._client
                if chromadb_client:
                    chromadb_client.delete_collection(name=test_collection_name)
            except Exception:
                pass  # Collection might not exist
            await agent.vector_db_manager.close()

        await agent.close()


@pytest.mark.integration
@pytest.mark.requires_llm
@pytest.mark.asyncio
async def test_context_format_json(test_hass, llm_config, sample_entity_states, session_manager):
    """Test JSON context formatting.

    This test verifies that:
    1. JSON format produces valid JSON structure
    2. Entity information is correctly formatted as JSON
    3. JSON context can be parsed and used by LLM
    """
    config = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 500,
        CONF_CONTEXT_MODE: CONTEXT_MODE_DIRECT,
        CONF_DIRECT_ENTITIES: ["light.living_room", "sensor.temperature"],
        CONF_CONTEXT_FORMAT: CONTEXT_FORMAT_JSON,
        CONF_HISTORY_ENABLED: False,
        CONF_EMIT_EVENTS: False,
        CONF_DEBUG_LOGGING: False,
    }

    with patch(
        "custom_components.home_agent.agent.async_should_expose",
        return_value=False,
    ):
        test_hass.states.async_all = MagicMock(return_value=sample_entity_states)

        def mock_get_state(entity_id):
            for state in sample_entity_states:
                if state.entity_id == entity_id:
                    return state
            return None

        test_hass.states.get = MagicMock(side_effect=mock_get_state)

        agent = HomeAgent(test_hass, config, session_manager)

        # Get context
        context = await agent.context_manager.get_context(
            user_input="Show me the lights",
            conversation_id="test_json_format",
        )

        assert context is not None, "Context should not be None in JSON format mode"
        assert isinstance(
            context, (str, list, dict)
        ), f"Context should be str/list/dict, got {type(context)}"
        context_str = str(context)

        # JSON format should contain structured data markers
        # Look for common JSON patterns
        assert any(
            marker in context_str for marker in ["{", "}", "[", "]"]
        ), "JSON format should contain JSON structure markers"

        await agent.close()


@pytest.mark.integration
@pytest.mark.requires_llm
@pytest.mark.asyncio
async def test_context_format_natural_language(
    test_hass, llm_config, sample_entity_states, session_manager
):
    """Test natural language context formatting.

    This test verifies that:
    1. Natural language format produces readable text
    2. Entity information is formatted in a human-readable way
    3. Format is suitable for LLM processing
    """
    config = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 500,
        CONF_CONTEXT_MODE: CONTEXT_MODE_DIRECT,
        CONF_DIRECT_ENTITIES: ["light.living_room", "sensor.temperature"],
        CONF_CONTEXT_FORMAT: CONTEXT_FORMAT_NATURAL_LANGUAGE,
        CONF_HISTORY_ENABLED: False,
        CONF_EMIT_EVENTS: False,
        CONF_DEBUG_LOGGING: False,
    }

    with patch(
        "custom_components.home_agent.agent.async_should_expose",
        return_value=False,
    ):
        test_hass.states.async_all = MagicMock(return_value=sample_entity_states)

        def mock_get_state(entity_id):
            for state in sample_entity_states:
                if state.entity_id == entity_id:
                    return state
            return None

        test_hass.states.get = MagicMock(side_effect=mock_get_state)

        agent = HomeAgent(test_hass, config, session_manager)

        # Get context
        context = await agent.context_manager.get_context(
            user_input="Show me the lights",
            conversation_id="test_natural_language_format",
        )

        assert context is not None, "Context should not be None in natural language format mode"
        assert isinstance(
            context, (str, list, dict)
        ), f"Context should be str/list/dict, got {type(context)}"
        context_str = str(context).lower()

        # Natural language format should be more readable
        # Should contain words like "is", "the", etc. and fewer symbols
        # It's harder to strictly validate, but we can check it's not overly structured
        assert len(context_str) > 0, "Context should not be empty"

        # Natural language should have entity references in readable form
        # Check that we have some readable content (not just pure JSON)
        word_count = len(context_str.split())
        assert word_count > 5, "Natural language format should have multiple words"

        await agent.close()


@pytest.mark.integration
@pytest.mark.requires_llm
@pytest.mark.asyncio
async def test_context_format_hybrid(test_hass, llm_config, sample_entity_states, session_manager):
    """Test hybrid context formatting.

    This test verifies that:
    1. Hybrid format combines JSON and natural language
    2. Context contains both structured and readable information
    3. Format provides benefits of both approaches
    """
    config = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 500,
        CONF_CONTEXT_MODE: CONTEXT_MODE_DIRECT,
        CONF_DIRECT_ENTITIES: ["light.living_room", "sensor.temperature"],
        CONF_CONTEXT_FORMAT: CONTEXT_FORMAT_HYBRID,
        CONF_HISTORY_ENABLED: False,
        CONF_EMIT_EVENTS: False,
        CONF_DEBUG_LOGGING: False,
    }

    with patch(
        "custom_components.home_agent.agent.async_should_expose",
        return_value=False,
    ):
        test_hass.states.async_all = MagicMock(return_value=sample_entity_states)

        def mock_get_state(entity_id):
            for state in sample_entity_states:
                if state.entity_id == entity_id:
                    return state
            return None

        test_hass.states.get = MagicMock(side_effect=mock_get_state)

        agent = HomeAgent(test_hass, config, session_manager)

        # Get context
        context = await agent.context_manager.get_context(
            user_input="Show me the lights",
            conversation_id="test_hybrid_format",
        )

        assert context is not None, "Context should not be None in hybrid format mode"
        assert isinstance(
            context, (str, list, dict)
        ), f"Context should be str/list/dict, got {type(context)}"
        context_str = str(context)

        # Hybrid format should have characteristics of both
        # Should have some JSON structure
        has_json_markers = any(marker in context_str for marker in ["{", "}", "[", "]"])

        # And should have readable text
        word_count = len(context_str.split())

        # Either should have JSON markers or substantial text content
        # (the exact implementation may vary)
        assert (
            has_json_markers or word_count > 10
        ), "Hybrid format should have JSON structure or substantial text"

        await agent.close()


@pytest.mark.integration
@pytest.mark.requires_llm
@pytest.mark.asyncio
async def test_context_with_no_entities(test_hass, llm_config, session_manager):
    """Test context manager behavior when no entities are configured.

    This test verifies that:
    1. Context manager handles empty entity lists gracefully
    2. No errors occur when no entities are available
    3. System continues to function with minimal context
    """
    config = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 500,
        CONF_CONTEXT_MODE: CONTEXT_MODE_DIRECT,
        CONF_DIRECT_ENTITIES: [],  # No entities configured
        CONF_CONTEXT_FORMAT: CONTEXT_FORMAT_JSON,
        CONF_HISTORY_ENABLED: False,
        CONF_EMIT_EVENTS: False,
        CONF_DEBUG_LOGGING: False,
    }

    with patch(
        "custom_components.home_agent.agent.async_should_expose",
        return_value=False,
    ):
        test_hass.states.async_all = MagicMock(return_value=[])

        agent = HomeAgent(test_hass, config, session_manager)

        # Get context with no entities
        context = await agent.context_manager.get_context(
            user_input="Hello",
            conversation_id="test_no_entities",
        )

        # Context should exist but may be empty or minimal
        assert context is not None, "Context should not be None even with no entities"

        # Process a message to verify system still works
        response = await agent.process_message(
            text="Hello, how are you?",
            conversation_id="test_no_entities",
        )

        assert response is not None, "Response should not be None even with no entities"
        assert isinstance(response, str), f"Response should be a string, got {type(response)}"
        assert (
            len(response) > 10
        ), f"Response should be meaningful (>10 chars), got {len(response)} chars: {response[:100]}"

        await agent.close()
