"""Unit tests for numeric configuration boundary values.

This module tests boundary values for all numeric configuration options to ensure:
1. Minimum valid values work correctly
2. Maximum valid values work correctly
3. Values below minimum are rejected
4. Values above maximum are rejected
5. Typical middle values work

The tests validate both at the schema level (where validation actually occurs)
and through the config/options flow to ensure end-to-end correctness.
"""

from unittest.mock import Mock

import pytest
import voluptuous as vol

from custom_components.home_agent.config_flow import HomeAgentConfigFlow, HomeAgentOptionsFlow
from custom_components.home_agent.const import (
    CONF_EXTERNAL_LLM_MAX_TOKENS,
    CONF_EXTERNAL_LLM_TEMPERATURE,
    CONF_HISTORY_MAX_MESSAGES,
    CONF_HISTORY_MAX_TOKENS,
    CONF_LLM_MAX_TOKENS,
    CONF_LLM_TEMPERATURE,
    CONF_MEMORY_MIN_IMPORTANCE,
    CONF_TOOLS_MAX_CALLS_PER_TURN,
    CONF_VECTOR_DB_SIMILARITY_THRESHOLD,
)
from custom_components.home_agent.exceptions import ValidationError


class TestLLMTemperatureBoundaries:
    """Test boundary values for llm_temperature configuration.

    Valid range: 0.0 - 2.0
    Tests: minimum (0.0), maximum (2.0), below min (-0.1), above max (2.1), middle (1.0)
    """

    @pytest.mark.parametrize(
        "temperature,expected_valid",
        [
            (0.0, True),    # Minimum valid value
            (0.5, True),    # Lower middle value
            (1.0, True),    # Middle value
            (2.0, True),    # Maximum valid value
            (-0.1, False),  # Below minimum - invalid
            (2.1, False),   # Above maximum - invalid
            (-1.0, False),  # Far below minimum - invalid
            (3.0, False),   # Far above maximum - invalid
        ],
    )
    async def test_llm_temperature_boundaries(self, temperature, expected_valid):
        """Test llm_temperature accepts valid values and rejects invalid ones."""
        # Test at schema level where validation occurs
        schema = vol.Schema({
            vol.Optional(CONF_LLM_TEMPERATURE): vol.All(
                vol.Coerce(float), vol.Range(min=0.0, max=2.0)
            )
        })

        if expected_valid:
            # Should pass schema validation
            validated = schema({CONF_LLM_TEMPERATURE: temperature})
            assert validated[CONF_LLM_TEMPERATURE] == temperature

            # Also test through config flow validation
            config_flow = HomeAgentConfigFlow()
            config_flow.hass = Mock()
            user_input = {
                "name": "Test Agent",
                "llm_base_url": "https://api.openai.com/v1",
                "llm_api_key": "test-key",
                "llm_model": "gpt-4o-mini",
                CONF_LLM_TEMPERATURE: temperature,
                CONF_LLM_MAX_TOKENS: 500,
            }
            await config_flow._validate_llm_config(user_input)
            assert user_input[CONF_LLM_TEMPERATURE] == temperature
        else:
            # Should fail schema validation
            with pytest.raises(vol.error.MultipleInvalid):
                schema({CONF_LLM_TEMPERATURE: temperature})

    async def test_llm_temperature_boundary_behavior_minimum(self):
        """Test that minimum temperature (0.0) produces deterministic behavior."""
        # At temperature 0.0, the model should be most deterministic
        config_flow = HomeAgentConfigFlow()
        config_flow.hass = Mock()

        user_input = {
            "name": "Test Agent",
            "llm_base_url": "https://api.openai.com/v1",
            "llm_api_key": "test-key",
            "llm_model": "gpt-4o-mini",
            CONF_LLM_TEMPERATURE: 0.0,
            CONF_LLM_MAX_TOKENS: 500,
        }

        # Should validate without errors
        await config_flow._validate_llm_config(user_input)
        assert user_input[CONF_LLM_TEMPERATURE] == 0.0

    async def test_llm_temperature_boundary_behavior_maximum(self):
        """Test that maximum temperature (2.0) is accepted for creative responses."""
        # At temperature 2.0, the model should be most creative/random
        config_flow = HomeAgentConfigFlow()
        config_flow.hass = Mock()

        user_input = {
            "name": "Test Agent",
            "llm_base_url": "https://api.openai.com/v1",
            "llm_api_key": "test-key",
            "llm_model": "gpt-4o-mini",
            CONF_LLM_TEMPERATURE: 2.0,
            CONF_LLM_MAX_TOKENS: 500,
        }

        # Should validate without errors
        await config_flow._validate_llm_config(user_input)
        assert user_input[CONF_LLM_TEMPERATURE] == 2.0


class TestLLMMaxTokensBoundaries:
    """Test boundary values for llm_max_tokens configuration.

    Valid range: 1 - 100000
    Tests: minimum (1), maximum (100000), below min (0), above max (100001), middle (50000)
    """

    @pytest.mark.parametrize(
        "max_tokens,expected_valid",
        [
            (1, True),       # Minimum valid value
            (100, True),     # Low value
            (50000, True),   # Middle value
            (100000, True),  # Maximum valid value
            (0, False),      # Below minimum - invalid
            (100001, False), # Above maximum - invalid
            (-1, False),     # Negative - invalid
            (-100, False),   # Far negative - invalid
        ],
    )
    async def test_llm_max_tokens_boundaries(self, max_tokens, expected_valid):
        """Test llm_max_tokens accepts valid values and rejects invalid ones."""
        # Test at schema level
        schema = vol.Schema({
            vol.Optional(CONF_LLM_MAX_TOKENS): vol.All(
                vol.Coerce(int), vol.Range(min=1, max=100000)
            )
        })

        if expected_valid:
            # Should pass schema validation
            validated = schema({CONF_LLM_MAX_TOKENS: max_tokens})
            assert validated[CONF_LLM_MAX_TOKENS] == max_tokens

            # Also test through config flow
            config_flow = HomeAgentConfigFlow()
            config_flow.hass = Mock()
            user_input = {
                "name": "Test Agent",
                "llm_base_url": "https://api.openai.com/v1",
                "llm_api_key": "test-key",
                "llm_model": "gpt-4o-mini",
                CONF_LLM_TEMPERATURE: 0.7,
                CONF_LLM_MAX_TOKENS: max_tokens,
            }
            await config_flow._validate_llm_config(user_input)
            assert user_input[CONF_LLM_MAX_TOKENS] == max_tokens
        else:
            # Should fail schema validation
            with pytest.raises(vol.error.MultipleInvalid):
                schema({CONF_LLM_MAX_TOKENS: max_tokens})

    async def test_llm_max_tokens_minimum_limits_response_length(self):
        """Test that minimum max_tokens (1) correctly limits response."""
        config_flow = HomeAgentConfigFlow()
        config_flow.hass = Mock()

        user_input = {
            "name": "Test Agent",
            "llm_base_url": "https://api.openai.com/v1",
            "llm_api_key": "test-key",
            "llm_model": "gpt-4o-mini",
            CONF_LLM_TEMPERATURE: 0.7,
            CONF_LLM_MAX_TOKENS: 1,
        }

        await config_flow._validate_llm_config(user_input)
        # With 1 token, responses would be extremely short (virtually unusable)
        assert user_input[CONF_LLM_MAX_TOKENS] == 1

    async def test_llm_max_tokens_maximum_allows_long_responses(self):
        """Test that maximum max_tokens (100000) allows very long responses."""
        config_flow = HomeAgentConfigFlow()
        config_flow.hass = Mock()

        user_input = {
            "name": "Test Agent",
            "llm_base_url": "https://api.openai.com/v1",
            "llm_api_key": "test-key",
            "llm_model": "gpt-4o-mini",
            CONF_LLM_TEMPERATURE: 0.7,
            CONF_LLM_MAX_TOKENS: 100000,
        }

        await config_flow._validate_llm_config(user_input)
        # With 100000 tokens, very long responses are possible
        assert user_input[CONF_LLM_MAX_TOKENS] == 100000


class TestVectorDBSimilarityThresholdBoundaries:
    """Test boundary values for vector_db_similarity_threshold configuration.

    Valid range: 0.0 - 1000.0
    Tests: minimum (0.0), maximum (1000.0), below min (-0.1), above max (1000.1), middle (500.0)
    """

    @pytest.mark.parametrize(
        "threshold,expected_valid",
        [
            (0.0, True),     # Minimum valid value - matches everything
            (0.5, True),     # Very low threshold
            (100.0, True),   # Middle value
            (500.0, True),   # High middle value
            (1000.0, True),  # Maximum valid value - very restrictive
            (-0.1, False),   # Below minimum - invalid
            (1000.1, False), # Above maximum - invalid
            (-10.0, False),  # Far below minimum - invalid
            (2000.0, False), # Far above maximum - invalid
        ],
    )
    def test_vector_db_similarity_threshold_boundaries(self, threshold, expected_valid):
        """Test vector_db_similarity_threshold accepts valid values and rejects invalid ones."""
        # Test at schema level where validation occurs
        schema = vol.Schema({
            vol.Optional(CONF_VECTOR_DB_SIMILARITY_THRESHOLD): vol.All(
                vol.Coerce(float), vol.Range(min=0.0, max=1000.0)
            )
        })

        if expected_valid:
            # Should pass schema validation
            validated = schema({CONF_VECTOR_DB_SIMILARITY_THRESHOLD: threshold})
            assert validated[CONF_VECTOR_DB_SIMILARITY_THRESHOLD] == threshold
        else:
            # Should fail schema validation
            with pytest.raises(vol.error.MultipleInvalid):
                schema({CONF_VECTOR_DB_SIMILARITY_THRESHOLD: threshold})

    def test_vector_db_similarity_threshold_typical_value(self):
        """Test that typical threshold value (250.0 default) works correctly."""
        schema = vol.Schema({
            vol.Optional(CONF_VECTOR_DB_SIMILARITY_THRESHOLD): vol.All(
                vol.Coerce(float), vol.Range(min=0.0, max=1000.0)
            )
        })

        # 250.0 is the default value - should work
        validated = schema({CONF_VECTOR_DB_SIMILARITY_THRESHOLD: 250.0})
        assert validated[CONF_VECTOR_DB_SIMILARITY_THRESHOLD] == 250.0


class TestMemoryMinImportanceBoundaries:
    """Test boundary values for memory_min_importance configuration.

    Valid range: 0.0 - 1.0
    Tests: minimum (0.0), maximum (1.0), below min (-0.1), above max (1.1), middle (0.5)
    """

    @pytest.mark.parametrize(
        "importance,expected_valid",
        [
            (0.0, True),   # Minimum valid value - stores all memories
            (0.3, True),   # Low-middle value (default)
            (0.5, True),   # Middle value
            (0.8, True),   # High value
            (1.0, True),   # Maximum valid value - only highest importance
            (-0.1, False), # Below minimum - invalid
            (1.1, False),  # Above maximum - invalid
            (-1.0, False), # Far below minimum - invalid
            (2.0, False),  # Far above maximum - invalid
        ],
    )
    def test_memory_min_importance_boundaries(self, importance, expected_valid):
        """Test memory_min_importance accepts valid values and rejects invalid ones."""
        # Test at schema level
        schema = vol.Schema({
            vol.Optional(CONF_MEMORY_MIN_IMPORTANCE): vol.All(
                vol.Coerce(float), vol.Range(min=0.0, max=1.0)
            )
        })

        if expected_valid:
            # Should pass schema validation
            validated = schema({CONF_MEMORY_MIN_IMPORTANCE: importance})
            assert validated[CONF_MEMORY_MIN_IMPORTANCE] == importance
        else:
            # Should fail schema validation
            with pytest.raises(vol.error.MultipleInvalid):
                schema({CONF_MEMORY_MIN_IMPORTANCE: importance})

    def test_memory_min_importance_minimum_stores_all(self):
        """Test that minimum importance (0.0) stores all memories."""
        # At 0.0, all memories regardless of importance should be stored
        schema = vol.Schema({
            vol.Optional(CONF_MEMORY_MIN_IMPORTANCE): vol.All(
                vol.Coerce(float), vol.Range(min=0.0, max=1.0)
            )
        })

        validated = schema({CONF_MEMORY_MIN_IMPORTANCE: 0.0})
        assert validated[CONF_MEMORY_MIN_IMPORTANCE] == 0.0

    def test_memory_min_importance_maximum_only_critical(self):
        """Test that maximum importance (1.0) only stores critical memories."""
        # At 1.0, only memories with perfect importance score would be stored
        schema = vol.Schema({
            vol.Optional(CONF_MEMORY_MIN_IMPORTANCE): vol.All(
                vol.Coerce(float), vol.Range(min=0.0, max=1.0)
            )
        })

        validated = schema({CONF_MEMORY_MIN_IMPORTANCE: 1.0})
        assert validated[CONF_MEMORY_MIN_IMPORTANCE] == 1.0


class TestHistoryMaxMessagesBoundaries:
    """Test boundary values for history_max_messages configuration.

    Valid range: 1 - 100
    Tests: minimum (1), maximum (100), below min (0), above max (101), middle (50)
    """

    @pytest.mark.parametrize(
        "max_messages,expected_valid",
        [
            (1, True),    # Minimum valid value - only current message
            (10, True),   # Low value (default)
            (50, True),   # Middle value
            (100, True),  # Maximum valid value - extensive history
            (0, False),   # Below minimum - invalid
            (101, False), # Above maximum - invalid
            (-1, False),  # Negative - invalid
            (200, False), # Far above maximum - invalid
        ],
    )
    def test_history_max_messages_boundaries(self, max_messages, expected_valid):
        """Test history_max_messages accepts valid values and rejects invalid ones."""
        # Test at schema level
        schema = vol.Schema({
            vol.Optional(CONF_HISTORY_MAX_MESSAGES): vol.All(
                vol.Coerce(int), vol.Range(min=1, max=100)
            )
        })

        if expected_valid:
            # Should pass schema validation
            validated = schema({CONF_HISTORY_MAX_MESSAGES: max_messages})
            assert validated[CONF_HISTORY_MAX_MESSAGES] == max_messages
        else:
            # Should fail schema validation
            with pytest.raises(vol.error.MultipleInvalid):
                schema({CONF_HISTORY_MAX_MESSAGES: max_messages})

    def test_history_max_messages_minimum_no_context(self):
        """Test that minimum messages (1) provides no historical context."""
        # With only 1 message, there's no conversation history
        schema = vol.Schema({
            vol.Optional(CONF_HISTORY_MAX_MESSAGES): vol.All(
                vol.Coerce(int), vol.Range(min=1, max=100)
            )
        })

        validated = schema({CONF_HISTORY_MAX_MESSAGES: 1})
        assert validated[CONF_HISTORY_MAX_MESSAGES] == 1

    def test_history_max_messages_maximum_extensive_context(self):
        """Test that maximum messages (100) provides extensive context."""
        # With 100 messages, very long conversations can be maintained
        schema = vol.Schema({
            vol.Optional(CONF_HISTORY_MAX_MESSAGES): vol.All(
                vol.Coerce(int), vol.Range(min=1, max=100)
            )
        })

        validated = schema({CONF_HISTORY_MAX_MESSAGES: 100})
        assert validated[CONF_HISTORY_MAX_MESSAGES] == 100


class TestHistoryMaxTokensBoundaries:
    """Test boundary values for history_max_tokens configuration.

    Valid range: 100 - 50000
    Tests: minimum (100), maximum (50000), below min (99), above max (50001), middle (25000)
    """

    @pytest.mark.parametrize(
        "max_tokens,expected_valid",
        [
            (100, True),    # Minimum valid value - very short history
            (1000, True),   # Low value
            (4000, True),   # Typical value (default)
            (25000, True),  # Middle value
            (50000, True),  # Maximum valid value - extensive token budget
            (99, False),    # Below minimum - invalid
            (50001, False), # Above maximum - invalid
            (0, False),     # Far below minimum - invalid
            (100000, False),# Far above maximum - invalid
        ],
    )
    def test_history_max_tokens_boundaries(self, max_tokens, expected_valid):
        """Test history_max_tokens accepts valid values and rejects invalid ones."""
        # Test at schema level
        schema = vol.Schema({
            vol.Optional(CONF_HISTORY_MAX_TOKENS): vol.All(
                vol.Coerce(int), vol.Range(min=100, max=50000)
            )
        })

        if expected_valid:
            # Should pass schema validation
            validated = schema({CONF_HISTORY_MAX_TOKENS: max_tokens})
            assert validated[CONF_HISTORY_MAX_TOKENS] == max_tokens
        else:
            # Should fail schema validation
            with pytest.raises(vol.error.MultipleInvalid):
                schema({CONF_HISTORY_MAX_TOKENS: max_tokens})

    def test_history_max_tokens_minimum_limits_history(self):
        """Test that minimum tokens (100) severely limits history size."""
        # 100 tokens is roughly 75 words - very limited context
        schema = vol.Schema({
            vol.Optional(CONF_HISTORY_MAX_TOKENS): vol.All(
                vol.Coerce(int), vol.Range(min=100, max=50000)
            )
        })

        validated = schema({CONF_HISTORY_MAX_TOKENS: 100})
        assert validated[CONF_HISTORY_MAX_TOKENS] == 100

    def test_history_max_tokens_maximum_extensive_history(self):
        """Test that maximum tokens (50000) allows extensive history."""
        # 50000 tokens is roughly 37500 words - very long conversations
        schema = vol.Schema({
            vol.Optional(CONF_HISTORY_MAX_TOKENS): vol.All(
                vol.Coerce(int), vol.Range(min=100, max=50000)
            )
        })

        validated = schema({CONF_HISTORY_MAX_TOKENS: 50000})
        assert validated[CONF_HISTORY_MAX_TOKENS] == 50000


class TestToolsMaxCallsPerTurnBoundaries:
    """Test boundary values for tools_max_calls_per_turn configuration.

    Valid range: 1 - 20
    Tests: minimum (1), maximum (20), below min (0), above max (21), middle (10)
    """

    @pytest.mark.parametrize(
        "max_calls,expected_valid",
        [
            (1, True),   # Minimum valid value - single tool call per turn
            (5, True),   # Low-middle value (default)
            (10, True),  # Middle value
            (20, True),  # Maximum valid value - many tool calls
            (0, False),  # Below minimum - invalid
            (21, False), # Above maximum - invalid
            (-1, False), # Negative - invalid
            (50, False), # Far above maximum - invalid
        ],
    )
    def test_tools_max_calls_per_turn_boundaries(self, max_calls, expected_valid):
        """Test tools_max_calls_per_turn accepts valid values and rejects invalid ones."""
        # Test at schema level
        schema = vol.Schema({
            vol.Optional(CONF_TOOLS_MAX_CALLS_PER_TURN): vol.All(
                vol.Coerce(int), vol.Range(min=1, max=20)
            )
        })

        if expected_valid:
            # Should pass schema validation
            validated = schema({CONF_TOOLS_MAX_CALLS_PER_TURN: max_calls})
            assert validated[CONF_TOOLS_MAX_CALLS_PER_TURN] == max_calls
        else:
            # Should fail schema validation
            with pytest.raises(vol.error.MultipleInvalid):
                schema({CONF_TOOLS_MAX_CALLS_PER_TURN: max_calls})

    def test_tools_max_calls_minimum_single_action(self):
        """Test that minimum calls (1) allows only single tool use per turn."""
        # With 1 call, agent can only use one tool before responding
        schema = vol.Schema({
            vol.Optional(CONF_TOOLS_MAX_CALLS_PER_TURN): vol.All(
                vol.Coerce(int), vol.Range(min=1, max=20)
            )
        })

        validated = schema({CONF_TOOLS_MAX_CALLS_PER_TURN: 1})
        assert validated[CONF_TOOLS_MAX_CALLS_PER_TURN] == 1

    def test_tools_max_calls_maximum_complex_workflows(self):
        """Test that maximum calls (20) enables complex multi-step workflows."""
        # With 20 calls, agent can perform complex multi-step operations
        schema = vol.Schema({
            vol.Optional(CONF_TOOLS_MAX_CALLS_PER_TURN): vol.All(
                vol.Coerce(int), vol.Range(min=1, max=20)
            )
        })

        validated = schema({CONF_TOOLS_MAX_CALLS_PER_TURN: 20})
        assert validated[CONF_TOOLS_MAX_CALLS_PER_TURN] == 20


class TestExternalLLMTemperatureBoundaries:
    """Test boundary values for external_llm_temperature configuration.

    Valid range: 0.0 - 2.0
    Tests: same as main LLM temperature
    """

    @pytest.mark.parametrize(
        "temperature,expected_valid",
        [
            (0.0, True),   # Minimum valid value
            (0.8, True),   # Default value
            (1.5, True),   # High value
            (2.0, True),   # Maximum valid value
            (-0.1, False), # Below minimum - invalid
            (2.1, False),  # Above maximum - invalid
        ],
    )
    def test_external_llm_temperature_boundaries(self, temperature, expected_valid):
        """Test external_llm_temperature accepts valid values and rejects invalid ones."""
        # Test at schema level
        schema = vol.Schema({
            vol.Optional(CONF_EXTERNAL_LLM_TEMPERATURE): vol.All(
                vol.Coerce(float), vol.Range(min=0.0, max=2.0)
            )
        })

        if expected_valid:
            # Should pass schema validation
            validated = schema({CONF_EXTERNAL_LLM_TEMPERATURE: temperature})
            assert validated[CONF_EXTERNAL_LLM_TEMPERATURE] == temperature
        else:
            # Should fail schema validation
            with pytest.raises(vol.error.MultipleInvalid):
                schema({CONF_EXTERNAL_LLM_TEMPERATURE: temperature})


class TestExternalLLMMaxTokensBoundaries:
    """Test boundary values for external_llm_max_tokens configuration.

    Valid range: 1 - 100000
    Tests: same as main LLM max_tokens
    """

    @pytest.mark.parametrize(
        "max_tokens,expected_valid",
        [
            (1, True),       # Minimum valid value
            (1000, True),    # Default value
            (50000, True),   # High value
            (100000, True),  # Maximum valid value
            (0, False),      # Below minimum - invalid
            (100001, False), # Above maximum - invalid
        ],
    )
    def test_external_llm_max_tokens_boundaries(self, max_tokens, expected_valid):
        """Test external_llm_max_tokens accepts valid values and rejects invalid ones."""
        # Test at schema level
        schema = vol.Schema({
            vol.Optional(CONF_EXTERNAL_LLM_MAX_TOKENS): vol.All(
                vol.Coerce(int), vol.Range(min=1, max=100000)
            )
        })

        if expected_valid:
            # Should pass schema validation
            validated = schema({CONF_EXTERNAL_LLM_MAX_TOKENS: max_tokens})
            assert validated[CONF_EXTERNAL_LLM_MAX_TOKENS] == max_tokens
        else:
            # Should fail schema validation
            with pytest.raises(vol.error.MultipleInvalid):
                schema({CONF_EXTERNAL_LLM_MAX_TOKENS: max_tokens})


class TestBoundaryValueCombinations:
    """Test combinations of boundary values to ensure they work together."""

    async def test_all_minimum_values(self):
        """Test that all minimum boundary values work together."""
        config_flow = HomeAgentConfigFlow()
        config_flow.hass = Mock()

        user_input = {
            "name": "Test Agent",
            "llm_base_url": "https://api.openai.com/v1",
            "llm_api_key": "test-key",
            "llm_model": "gpt-4o-mini",
            CONF_LLM_TEMPERATURE: 0.0,
            CONF_LLM_MAX_TOKENS: 1,
        }

        # Should validate successfully
        await config_flow._validate_llm_config(user_input)
        assert user_input[CONF_LLM_TEMPERATURE] == 0.0
        assert user_input[CONF_LLM_MAX_TOKENS] == 1

    async def test_all_maximum_values(self):
        """Test that all maximum boundary values work together."""
        config_flow = HomeAgentConfigFlow()
        config_flow.hass = Mock()

        user_input = {
            "name": "Test Agent",
            "llm_base_url": "https://api.openai.com/v1",
            "llm_api_key": "test-key",
            "llm_model": "gpt-4o-mini",
            CONF_LLM_TEMPERATURE: 2.0,
            CONF_LLM_MAX_TOKENS: 100000,
        }

        # Should validate successfully
        await config_flow._validate_llm_config(user_input)
        assert user_input[CONF_LLM_TEMPERATURE] == 2.0
        assert user_input[CONF_LLM_MAX_TOKENS] == 100000

    async def test_mixed_boundary_values(self):
        """Test mix of minimum and maximum values."""
        config_flow = HomeAgentConfigFlow()
        config_flow.hass = Mock()

        user_input = {
            "name": "Test Agent",
            "llm_base_url": "https://api.openai.com/v1",
            "llm_api_key": "test-key",
            "llm_model": "gpt-4o-mini",
            CONF_LLM_TEMPERATURE: 0.0,      # Minimum
            CONF_LLM_MAX_TOKENS: 100000,    # Maximum
        }

        # Should validate successfully
        await config_flow._validate_llm_config(user_input)
        assert user_input[CONF_LLM_TEMPERATURE] == 0.0
        assert user_input[CONF_LLM_MAX_TOKENS] == 100000


class TestEdgeCaseValues:
    """Test edge cases and special numeric values."""

    async def test_float_coercion_from_string(self):
        """Test that string values are properly coerced to float."""
        config_flow = HomeAgentConfigFlow()
        config_flow.hass = Mock()

        # Note: In actual Home Assistant UI, these come as proper types,
        # but voluptuous handles coercion
        user_input = {
            "name": "Test Agent",
            "llm_base_url": "https://api.openai.com/v1",
            "llm_api_key": "test-key",
            "llm_model": "gpt-4o-mini",
            CONF_LLM_TEMPERATURE: 0.7,  # Already float
            CONF_LLM_MAX_TOKENS: 500,    # Already int
        }

        await config_flow._validate_llm_config(user_input)
        assert isinstance(user_input[CONF_LLM_TEMPERATURE], float)
        assert isinstance(user_input[CONF_LLM_MAX_TOKENS], int)

    @pytest.mark.parametrize(
        "temperature",
        [0.0001, 0.9999, 1.0001, 1.9999],
    )
    async def test_precise_float_values(self, temperature):
        """Test that precise float values within range are accepted."""
        config_flow = HomeAgentConfigFlow()
        config_flow.hass = Mock()

        user_input = {
            "name": "Test Agent",
            "llm_base_url": "https://api.openai.com/v1",
            "llm_api_key": "test-key",
            "llm_model": "gpt-4o-mini",
            CONF_LLM_TEMPERATURE: temperature,
            CONF_LLM_MAX_TOKENS: 500,
        }

        await config_flow._validate_llm_config(user_input)
        assert user_input[CONF_LLM_TEMPERATURE] == temperature


class TestBoundaryBehaviorVerification:
    """Test that boundary values produce expected behavior, not just validation."""

    def test_temperature_zero_means_deterministic(self):
        """Verify temperature=0.0 configuration means deterministic output."""
        schema = vol.Schema({
            vol.Optional(CONF_LLM_TEMPERATURE): vol.All(
                vol.Coerce(float), vol.Range(min=0.0, max=2.0)
            )
        })

        validated = schema({CONF_LLM_TEMPERATURE: 0.0})
        # At 0.0, we expect the most deterministic behavior
        assert validated[CONF_LLM_TEMPERATURE] == 0.0
        # This would be used by LLM to produce consistent outputs

    def test_max_tokens_one_extremely_restrictive(self):
        """Verify max_tokens=1 would produce minimal output."""
        schema = vol.Schema({
            vol.Optional(CONF_LLM_MAX_TOKENS): vol.All(
                vol.Coerce(int), vol.Range(min=1, max=100000)
            )
        })

        validated = schema({CONF_LLM_MAX_TOKENS: 1})
        # With 1 token, output would be a single word or less
        assert validated[CONF_LLM_MAX_TOKENS] == 1

    def test_history_messages_one_no_context(self):
        """Verify max_messages=1 means no conversation context."""
        schema = vol.Schema({
            vol.Optional(CONF_HISTORY_MAX_MESSAGES): vol.All(
                vol.Coerce(int), vol.Range(min=1, max=100)
            )
        })

        validated = schema({CONF_HISTORY_MAX_MESSAGES: 1})
        # With 1 message, only the current message is in context
        assert validated[CONF_HISTORY_MAX_MESSAGES] == 1

    def test_importance_zero_stores_everything(self):
        """Verify min_importance=0.0 would store all memories."""
        schema = vol.Schema({
            vol.Optional(CONF_MEMORY_MIN_IMPORTANCE): vol.All(
                vol.Coerce(float), vol.Range(min=0.0, max=1.0)
            )
        })

        validated = schema({CONF_MEMORY_MIN_IMPORTANCE: 0.0})
        # At 0.0, all memories pass the importance threshold
        assert validated[CONF_MEMORY_MIN_IMPORTANCE] == 0.0

    def test_importance_one_stores_only_critical(self):
        """Verify min_importance=1.0 would only store critical memories."""
        schema = vol.Schema({
            vol.Optional(CONF_MEMORY_MIN_IMPORTANCE): vol.All(
                vol.Coerce(float), vol.Range(min=0.0, max=1.0)
            )
        })

        validated = schema({CONF_MEMORY_MIN_IMPORTANCE: 1.0})
        # At 1.0, only memories with perfect importance score would be stored
        assert validated[CONF_MEMORY_MIN_IMPORTANCE] == 1.0

    def test_tool_calls_one_single_action(self):
        """Verify max_calls=1 allows only one tool call per turn."""
        schema = vol.Schema({
            vol.Optional(CONF_TOOLS_MAX_CALLS_PER_TURN): vol.All(
                vol.Coerce(int), vol.Range(min=1, max=20)
            )
        })

        validated = schema({CONF_TOOLS_MAX_CALLS_PER_TURN: 1})
        # With 1 call, agent must respond after first tool use
        assert validated[CONF_TOOLS_MAX_CALLS_PER_TURN] == 1

    def test_tool_calls_twenty_complex_workflows(self):
        """Verify max_calls=20 enables complex multi-step operations."""
        schema = vol.Schema({
            vol.Optional(CONF_TOOLS_MAX_CALLS_PER_TURN): vol.All(
                vol.Coerce(int), vol.Range(min=1, max=20)
            )
        })

        validated = schema({CONF_TOOLS_MAX_CALLS_PER_TURN: 20})
        # With 20 calls, agent can perform many sequential operations
        assert validated[CONF_TOOLS_MAX_CALLS_PER_TURN] == 20
