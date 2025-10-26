"""Unit tests for the HomeAssistantQueryTool."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from homeassistant.core import State
from homeassistant.util import dt as dt_util

from custom_components.home_agent.tools.ha_query import HomeAssistantQueryTool
from custom_components.home_agent.const import (
    HISTORY_AGGREGATE_AVG,
    HISTORY_AGGREGATE_COUNT,
    HISTORY_AGGREGATE_MAX,
    HISTORY_AGGREGATE_MIN,
    HISTORY_AGGREGATE_SUM,
    TOOL_HA_QUERY,
)
from custom_components.home_agent.exceptions import (
    PermissionDenied,
    ToolExecutionError,
    ValidationError,
)


class TestHomeAssistantQueryTool:
    """Test the HomeAssistantQueryTool class."""

    def test_tool_initialization(self, mock_hass):
        """Test that tool initializes correctly."""
        tool = HomeAssistantQueryTool(mock_hass)
        assert tool.hass == mock_hass
        assert tool._exposed_entities is None

    def test_tool_initialization_with_exposed_entities(self, mock_hass, exposed_entities):
        """Test initialization with exposed entities."""
        tool = HomeAssistantQueryTool(mock_hass, exposed_entities)
        assert tool._exposed_entities == exposed_entities

    def test_tool_name(self, mock_hass):
        """Test that tool name is correct."""
        tool = HomeAssistantQueryTool(mock_hass)
        assert tool.name == TOOL_HA_QUERY

    def test_tool_description(self, mock_hass):
        """Test that tool has a description."""
        tool = HomeAssistantQueryTool(mock_hass)
        assert isinstance(tool.description, str)
        assert len(tool.description) > 0
        assert "query" in tool.description.lower() or "get" in tool.description.lower()

    def test_tool_parameters_schema(self, mock_hass):
        """Test that parameter schema is valid."""
        tool = HomeAssistantQueryTool(mock_hass)
        params = tool.parameters

        assert params["type"] == "object"
        assert "properties" in params
        assert "required" in params

        # Check required parameters
        assert "entity_id" in params["required"]

        # Check properties
        assert "entity_id" in params["properties"]
        assert "attributes" in params["properties"]
        assert "history" in params["properties"]

        # Check history schema
        history = params["properties"]["history"]
        assert "properties" in history
        assert "duration" in history["properties"]
        assert "aggregate" in history["properties"]

    def test_to_openai_format(self, mock_hass):
        """Test conversion to OpenAI format."""
        tool = HomeAssistantQueryTool(mock_hass)
        openai_format = tool.to_openai_format()

        assert openai_format["type"] == "function"
        assert openai_format["function"]["name"] == TOOL_HA_QUERY
        assert "description" in openai_format["function"]
        assert "parameters" in openai_format["function"]

    @pytest.mark.asyncio
    async def test_query_single_entity(self, mock_hass, sample_light_state):
        """Test querying a single entity."""
        tool = HomeAssistantQueryTool(mock_hass)

        mock_hass.states.async_entity_ids.return_value = ["light.living_room"]
        mock_hass.states.get.return_value = sample_light_state

        result = await tool.execute(entity_id="light.living_room")

        assert result["success"] is True
        assert result["entity_id"] == "light.living_room"
        assert result["count"] == 1
        assert len(result["entities"]) == 1

        entity = result["entities"][0]
        assert entity["entity_id"] == "light.living_room"
        assert entity["state"] == "on"
        assert "attributes" in entity

    @pytest.mark.asyncio
    async def test_query_with_wildcard_domain(self, mock_hass, sample_light_state, sample_sensor_state):
        """Test querying with domain wildcard (light.*)."""
        tool = HomeAssistantQueryTool(mock_hass)

        # Mock entity IDs
        mock_hass.states.async_entity_ids.return_value = [
            "light.living_room",
            "light.bedroom",
            "sensor.temperature"
        ]

        # Mock state retrieval
        def get_state(entity_id):
            if entity_id.startswith("light"):
                return sample_light_state
            return sample_sensor_state

        mock_hass.states.get.side_effect = get_state

        result = await tool.execute(entity_id="light.*")

        assert result["success"] is True
        assert result["count"] == 2  # Only lights
        assert all("light." in e["entity_id"] for e in result["entities"])

    @pytest.mark.asyncio
    async def test_query_with_wildcard_entity(self, mock_hass, sample_light_state):
        """Test querying with entity name wildcard (*.living_room)."""
        tool = HomeAssistantQueryTool(mock_hass)

        mock_hass.states.async_entity_ids.return_value = [
            "light.living_room",
            "sensor.living_room_temperature",
            "climate.living_room_thermostat",
            "switch.bedroom"
        ]

        mock_hass.states.get.return_value = sample_light_state

        result = await tool.execute(entity_id="*.living_room*")

        assert result["success"] is True
        assert result["count"] > 0
        # All entities should have "living_room" in name
        for entity in result["entities"]:
            assert "living_room" in entity["entity_id"]

    @pytest.mark.asyncio
    async def test_query_with_attribute_filter(self, mock_hass, sample_light_state):
        """Test querying with specific attributes filter."""
        tool = HomeAssistantQueryTool(mock_hass)

        mock_hass.states.async_entity_ids.return_value = ["light.living_room"]
        mock_hass.states.get.return_value = sample_light_state

        result = await tool.execute(
            entity_id="light.living_room",
            attributes=["brightness", "color_temp"]
        )

        assert result["success"] is True
        entity = result["entities"][0]

        # Only requested attributes should be present
        assert "brightness" in entity["attributes"]
        assert "color_temp" in entity["attributes"]
        # rgb_color should not be included (not requested)
        assert "rgb_color" not in entity["attributes"]

    @pytest.mark.asyncio
    async def test_query_no_matches(self, mock_hass):
        """Test querying with no matching entities."""
        tool = HomeAssistantQueryTool(mock_hass)

        mock_hass.states.async_entity_ids.return_value = [
            "light.living_room",
            "sensor.temperature"
        ]

        result = await tool.execute(entity_id="switch.nonexistent")

        assert result["success"] is True
        assert result["count"] == 0
        assert result["entities"] == []
        assert "no entities found" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_query_missing_entity_id_raises_validation_error(self, mock_hass):
        """Test that missing entity_id raises ValidationError."""
        tool = HomeAssistantQueryTool(mock_hass)

        with pytest.raises(ValidationError) as exc_info:
            await tool.execute()

        assert "entity_id" in str(exc_info.value).lower()
        assert "required" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_invalid_entity_pattern_raises_validation_error(self, mock_hass):
        """Test that invalid entity pattern raises ValidationError."""
        tool = HomeAssistantQueryTool(mock_hass)

        # Missing dot
        with pytest.raises(ValidationError) as exc_info:
            await tool.execute(entity_id="invalid_pattern")

        assert "invalid entity_id pattern" in str(exc_info.value).lower()

    def test_is_valid_entity_pattern(self, mock_hass):
        """Test entity pattern validation."""
        tool = HomeAssistantQueryTool(mock_hass)

        # Valid patterns
        assert tool._is_valid_entity_pattern("light.living_room") is True
        assert tool._is_valid_entity_pattern("light.*") is True
        assert tool._is_valid_entity_pattern("*.living_room") is True
        assert tool._is_valid_entity_pattern("sensor.temp_*") is True

        # Invalid patterns
        assert tool._is_valid_entity_pattern("invalid") is False
        assert tool._is_valid_entity_pattern("too.many.dots") is False
        assert tool._is_valid_entity_pattern("invalid.CAPS") is False
        assert tool._is_valid_entity_pattern("") is False

    @pytest.mark.asyncio
    async def test_entity_access_validation_allowed(self, mock_hass, exposed_entities, sample_light_state):
        """Test entity access validation when entity is allowed."""
        tool = HomeAssistantQueryTool(mock_hass, exposed_entities)

        mock_hass.states.async_entity_ids.return_value = ["light.living_room"]
        mock_hass.states.get.return_value = sample_light_state

        # Should succeed - light.living_room is in exposed_entities
        result = await tool.execute(entity_id="light.living_room")
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_entity_access_validation_denied(self, mock_hass, exposed_entities):
        """Test entity access validation when entity is not allowed."""
        tool = HomeAssistantQueryTool(mock_hass, exposed_entities)

        mock_hass.states.async_entity_ids.return_value = ["light.secret_room"]

        with pytest.raises(PermissionDenied) as exc_info:
            await tool.execute(entity_id="light.secret_room")

        assert "not accessible" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_entity_access_validation_none_allows_all(self, mock_hass, sample_light_state):
        """Test that None exposed_entities allows all access."""
        tool = HomeAssistantQueryTool(mock_hass, exposed_entities=None)

        mock_hass.states.async_entity_ids.return_value = ["light.any_entity"]
        mock_hass.states.get.return_value = sample_light_state

        result = await tool.execute(entity_id="light.any_entity")
        assert result["success"] is True

    def test_format_entity_state(self, mock_hass, sample_light_state):
        """Test formatting entity state."""
        tool = HomeAssistantQueryTool(mock_hass)

        formatted = tool._format_entity_state(sample_light_state)

        assert formatted["entity_id"] == "light.living_room"
        assert formatted["state"] == "on"
        assert "last_changed" in formatted
        assert "last_updated" in formatted
        assert "attributes" in formatted
        assert formatted["attributes"]["brightness"] == 128

    def test_format_entity_state_with_attribute_filter(self, mock_hass, sample_light_state):
        """Test formatting entity state with attribute filter."""
        tool = HomeAssistantQueryTool(mock_hass)

        formatted = tool._format_entity_state(
            sample_light_state,
            attributes_filter=["brightness"]
        )

        assert "brightness" in formatted["attributes"]
        assert "color_temp" not in formatted["attributes"]

    @pytest.mark.asyncio
    async def test_query_history_with_duration(self, mock_hass, mock_history_states):
        """Test querying historical data."""
        tool = HomeAssistantQueryTool(mock_hass)

        mock_hass.states.async_entity_ids.return_value = ["sensor.living_room_temperature"]

        with patch("homeassistant.components.recorder.async_migration_in_progress") as mock_migration, \
             patch("homeassistant.components.recorder.async_recorder_ready") as mock_ready, \
             patch("homeassistant.components.recorder.history.state_changes_during_period") as mock_history:

            mock_migration.return_value = False
            mock_ready.return_value = True
            mock_history.return_value = {"sensor.living_room_temperature": mock_history_states}

            result = await tool.execute(
                entity_id="sensor.living_room_temperature",
                history={"duration": "24h"}
            )

            assert result["success"] is True
            assert "history" in result
            assert result["count"] == 1
            assert result["duration"] == "24h"

    @pytest.mark.asyncio
    async def test_query_history_with_avg_aggregation(self, mock_hass, mock_history_states):
        """Test querying historical data with average aggregation."""
        tool = HomeAssistantQueryTool(mock_hass)

        mock_hass.states.async_entity_ids.return_value = ["sensor.living_room_temperature"]

        with patch("homeassistant.components.recorder.async_migration_in_progress") as mock_migration, \
             patch("homeassistant.components.recorder.async_recorder_ready") as mock_ready, \
             patch("homeassistant.components.recorder.history.state_changes_during_period") as mock_history:

            mock_migration.return_value = False
            mock_ready.return_value = True
            mock_history.return_value = {"sensor.living_room_temperature": mock_history_states}

            result = await tool.execute(
                entity_id="sensor.living_room_temperature",
                history={"duration": "24h", "aggregate": HISTORY_AGGREGATE_AVG}
            )

            assert result["success"] is True
            assert result["aggregate"] == HISTORY_AGGREGATE_AVG
            history_data = result["history"][0]
            assert "value" in history_data
            assert isinstance(history_data["value"], (int, float))

    @pytest.mark.asyncio
    async def test_query_history_with_min_aggregation(self, mock_hass, mock_history_states):
        """Test querying historical data with min aggregation."""
        tool = HomeAssistantQueryTool(mock_hass)

        mock_hass.states.async_entity_ids.return_value = ["sensor.living_room_temperature"]

        with patch("homeassistant.components.recorder.async_migration_in_progress") as mock_migration, \
             patch("homeassistant.components.recorder.async_recorder_ready") as mock_ready, \
             patch("homeassistant.components.recorder.history.state_changes_during_period") as mock_history:

            mock_migration.return_value = False
            mock_ready.return_value = True
            mock_history.return_value = {"sensor.living_room_temperature": mock_history_states}

            result = await tool.execute(
                entity_id="sensor.living_room_temperature",
                history={"duration": "1h", "aggregate": HISTORY_AGGREGATE_MIN}
            )

            history_data = result["history"][0]
            assert history_data["aggregate"] == HISTORY_AGGREGATE_MIN
            # Min should be 68 based on mock data
            assert history_data["value"] == 68

    @pytest.mark.asyncio
    async def test_query_history_with_max_aggregation(self, mock_hass, mock_history_states):
        """Test querying historical data with max aggregation."""
        tool = HomeAssistantQueryTool(mock_hass)

        mock_hass.states.async_entity_ids.return_value = ["sensor.living_room_temperature"]

        with patch("homeassistant.components.recorder.async_migration_in_progress") as mock_migration, \
             patch("homeassistant.components.recorder.async_recorder_ready") as mock_ready, \
             patch("homeassistant.components.recorder.history.state_changes_during_period") as mock_history:

            mock_migration.return_value = False
            mock_ready.return_value = True
            mock_history.return_value = {"sensor.living_room_temperature": mock_history_states}

            result = await tool.execute(
                entity_id="sensor.living_room_temperature",
                history={"duration": "1h", "aggregate": HISTORY_AGGREGATE_MAX}
            )

            history_data = result["history"][0]
            assert history_data["aggregate"] == HISTORY_AGGREGATE_MAX
            # Max should be 77 based on mock data
            assert history_data["value"] == 77

    @pytest.mark.asyncio
    async def test_query_history_with_count_aggregation(self, mock_hass, mock_history_states):
        """Test querying historical data with count aggregation."""
        tool = HomeAssistantQueryTool(mock_hass)

        mock_hass.states.async_entity_ids.return_value = ["sensor.living_room_temperature"]

        with patch("homeassistant.components.recorder.async_migration_in_progress") as mock_migration, \
             patch("homeassistant.components.recorder.async_recorder_ready") as mock_ready, \
             patch("homeassistant.components.recorder.history.state_changes_during_period") as mock_history:

            mock_migration.return_value = False
            mock_ready.return_value = True
            mock_history.return_value = {"sensor.living_room_temperature": mock_history_states}

            result = await tool.execute(
                entity_id="sensor.living_room_temperature",
                history={"duration": "1h", "aggregate": HISTORY_AGGREGATE_COUNT}
            )

            history_data = result["history"][0]
            assert history_data["aggregate"] == HISTORY_AGGREGATE_COUNT
            assert history_data["value"] == len(mock_history_states)

    @pytest.mark.asyncio
    async def test_query_history_with_sum_aggregation(self, mock_hass, mock_history_states):
        """Test querying historical data with sum aggregation."""
        tool = HomeAssistantQueryTool(mock_hass)

        mock_hass.states.async_entity_ids.return_value = ["sensor.living_room_temperature"]

        with patch("homeassistant.components.recorder.async_migration_in_progress") as mock_migration, \
             patch("homeassistant.components.recorder.async_recorder_ready") as mock_ready, \
             patch("homeassistant.components.recorder.history.state_changes_during_period") as mock_history:

            mock_migration.return_value = False
            mock_ready.return_value = True
            mock_history.return_value = {"sensor.living_room_temperature": mock_history_states}

            result = await tool.execute(
                entity_id="sensor.living_room_temperature",
                history={"duration": "1h", "aggregate": HISTORY_AGGREGATE_SUM}
            )

            history_data = result["history"][0]
            assert history_data["aggregate"] == HISTORY_AGGREGATE_SUM
            assert isinstance(history_data["value"], (int, float))

    @pytest.mark.asyncio
    async def test_query_history_missing_duration_raises_validation_error(self, mock_hass):
        """Test that history query without duration raises ValidationError."""
        tool = HomeAssistantQueryTool(mock_hass)

        mock_hass.states.async_entity_ids.return_value = ["sensor.temperature"]

        with pytest.raises(ValidationError) as exc_info:
            await tool.execute(
                entity_id="sensor.temperature",
                history={"aggregate": HISTORY_AGGREGATE_AVG}
            )

        assert "duration" in str(exc_info.value).lower()
        assert "required" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_query_history_invalid_duration_format(self, mock_hass):
        """Test that invalid duration format raises ValidationError."""
        tool = HomeAssistantQueryTool(mock_hass)

        mock_hass.states.async_entity_ids.return_value = ["sensor.temperature"]

        with pytest.raises(ValidationError) as exc_info:
            await tool.execute(
                entity_id="sensor.temperature",
                history={"duration": "invalid"}
            )

        assert "invalid duration format" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_query_history_exceeds_max_duration(self, mock_hass):
        """Test that duration exceeding 30 days raises ValidationError."""
        tool = HomeAssistantQueryTool(mock_hass)

        mock_hass.states.async_entity_ids.return_value = ["sensor.temperature"]

        with pytest.raises(ValidationError) as exc_info:
            await tool.execute(
                entity_id="sensor.temperature",
                history={"duration": "31d"}
            )

        assert "exceeds maximum" in str(exc_info.value).lower()

    def test_parse_duration(self, mock_hass):
        """Test parsing duration strings."""
        tool = HomeAssistantQueryTool(mock_hass)

        # Valid durations
        assert tool._parse_duration("1h") == timedelta(hours=1)
        assert tool._parse_duration("24h") == timedelta(hours=24)
        assert tool._parse_duration("7d") == timedelta(days=7)
        assert tool._parse_duration("30m") == timedelta(minutes=30)
        assert tool._parse_duration("60s") == timedelta(seconds=60)

        # Invalid durations
        assert tool._parse_duration("invalid") is None
        assert tool._parse_duration("1w") is None  # weeks not supported
        assert tool._parse_duration("abc") is None

    def test_aggregate_history_avg(self, mock_hass, mock_history_states):
        """Test aggregating history with average."""
        tool = HomeAssistantQueryTool(mock_hass)

        result = tool._aggregate_history(mock_history_states, HISTORY_AGGREGATE_AVG)

        assert isinstance(result, float)
        # Average should be around 72.5 based on mock data
        assert 70 <= result <= 75

    def test_aggregate_history_min(self, mock_hass, mock_history_states):
        """Test aggregating history with minimum."""
        tool = HomeAssistantQueryTool(mock_hass)

        result = tool._aggregate_history(mock_history_states, HISTORY_AGGREGATE_MIN)

        assert result == 68  # Minimum value in mock data

    def test_aggregate_history_max(self, mock_hass, mock_history_states):
        """Test aggregating history with maximum."""
        tool = HomeAssistantQueryTool(mock_hass)

        result = tool._aggregate_history(mock_history_states, HISTORY_AGGREGATE_MAX)

        assert result == 77  # Maximum value in mock data

    def test_aggregate_history_count(self, mock_hass, mock_history_states):
        """Test aggregating history with count."""
        tool = HomeAssistantQueryTool(mock_hass)

        result = tool._aggregate_history(mock_history_states, HISTORY_AGGREGATE_COUNT)

        assert result == len(mock_history_states)

    def test_aggregate_history_sum(self, mock_hass, mock_history_states):
        """Test aggregating history with sum."""
        tool = HomeAssistantQueryTool(mock_hass)

        result = tool._aggregate_history(mock_history_states, HISTORY_AGGREGATE_SUM)

        assert isinstance(result, (int, float))
        assert result > 0

    def test_aggregate_history_empty_states(self, mock_hass):
        """Test aggregating empty state list."""
        tool = HomeAssistantQueryTool(mock_hass)

        result = tool._aggregate_history([], HISTORY_AGGREGATE_AVG)
        assert result is None

    def test_aggregate_history_non_numeric_states(self, mock_hass):
        """Test aggregating non-numeric states."""
        tool = HomeAssistantQueryTool(mock_hass)

        # Create states with non-numeric values
        states = [
            State("light.test", "on", attributes={}),
            State("light.test", "off", attributes={}),
        ]

        result = tool._aggregate_history(states, HISTORY_AGGREGATE_AVG)
        assert result is None  # No numeric values to aggregate

    def test_aggregate_history_unknown_aggregate_type(self, mock_hass, mock_history_states):
        """Test aggregating with unknown aggregate type."""
        tool = HomeAssistantQueryTool(mock_hass)

        result = tool._aggregate_history(mock_history_states, "unknown")
        assert result is None

    @pytest.mark.asyncio
    async def test_query_history_recorder_not_available(self, mock_hass):
        """Test that unavailable recorder raises ToolExecutionError."""
        tool = HomeAssistantQueryTool(mock_hass)

        mock_hass.states.async_entity_ids.return_value = ["sensor.temperature"]

        with patch("homeassistant.components.recorder.async_migration_in_progress") as mock_migration, \
             patch("homeassistant.components.recorder.async_recorder_ready") as mock_ready:

            mock_migration.return_value = False
            mock_ready.return_value = False  # Recorder not ready

            with pytest.raises(ToolExecutionError) as exc_info:
                await tool.execute(
                    entity_id="sensor.temperature",
                    history={"duration": "1h"}
                )

            assert "recorder" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_query_history_skips_failed_entities(self, mock_hass, mock_history_states):
        """Test that history query continues if one entity fails."""
        tool = HomeAssistantQueryTool(mock_hass)

        mock_hass.states.async_entity_ids.return_value = [
            "sensor.temperature1",
            "sensor.temperature2"
        ]

        with patch("homeassistant.components.recorder.async_migration_in_progress") as mock_migration, \
             patch("homeassistant.components.recorder.async_recorder_ready") as mock_ready, \
             patch("homeassistant.components.recorder.history.state_changes_during_period") as mock_history:

            mock_migration.return_value = False
            mock_ready.return_value = True

            # First entity succeeds, second fails
            call_count = [0]

            def history_side_effect(*args):
                call_count[0] += 1
                if call_count[0] == 1:
                    return {"sensor.temperature1": mock_history_states}
                else:
                    raise Exception("History query failed")

            mock_history.side_effect = history_side_effect

            result = await tool.execute(
                entity_id="sensor.temperature*",
                history={"duration": "1h"}
            )

            # Should succeed with one entity
            assert result["success"] is True
            assert result["count"] == 1

    def test_build_success_message(self, mock_hass):
        """Test building success messages."""
        tool = HomeAssistantQueryTool(mock_hass)

        # No entities found
        message = tool._build_success_message("light.*", 0)
        assert "no entities found" in message.lower()

        # One entity found
        message = tool._build_success_message("light.living_room", 1)
        assert "1 entity" in message.lower()

        # Multiple entities found
        message = tool._build_success_message("light.*", 5)
        assert "5 entities" in message.lower()

    @pytest.mark.asyncio
    async def test_find_matching_entities_exact_match(self, mock_hass):
        """Test finding entities with exact match."""
        tool = HomeAssistantQueryTool(mock_hass)

        mock_hass.states.async_entity_ids.return_value = [
            "light.living_room",
            "light.bedroom"
        ]

        matches = tool._find_matching_entities("light.living_room")

        assert len(matches) == 1
        assert "light.living_room" in matches

    @pytest.mark.asyncio
    async def test_find_matching_entities_wildcard(self, mock_hass):
        """Test finding entities with wildcard."""
        tool = HomeAssistantQueryTool(mock_hass)

        mock_hass.states.async_entity_ids.return_value = [
            "light.living_room",
            "light.bedroom",
            "sensor.temperature"
        ]

        matches = tool._find_matching_entities("light.*")

        assert len(matches) == 2
        assert "light.living_room" in matches
        assert "light.bedroom" in matches
        assert "sensor.temperature" not in matches

    @pytest.mark.asyncio
    async def test_find_matching_entities_no_matches(self, mock_hass):
        """Test finding entities with no matches."""
        tool = HomeAssistantQueryTool(mock_hass)

        mock_hass.states.async_entity_ids.return_value = [
            "light.living_room"
        ]

        matches = tool._find_matching_entities("switch.*")

        assert len(matches) == 0

    @pytest.mark.asyncio
    async def test_query_handles_exceptions_gracefully(self, mock_hass):
        """Test that unexpected exceptions are wrapped in ToolExecutionError."""
        tool = HomeAssistantQueryTool(mock_hass)

        mock_hass.states.async_entity_ids.side_effect = RuntimeError("Unexpected error")

        with pytest.raises(ToolExecutionError) as exc_info:
            await tool.execute(entity_id="light.living_room")

        assert "failed to query entities" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_response_includes_timestamps(self, mock_hass, sample_light_state):
        """Test that response includes last_changed and last_updated timestamps."""
        tool = HomeAssistantQueryTool(mock_hass)

        mock_hass.states.async_entity_ids.return_value = ["light.living_room"]
        mock_hass.states.get.return_value = sample_light_state

        result = await tool.execute(entity_id="light.living_room")

        entity = result["entities"][0]
        assert "last_changed" in entity
        assert "last_updated" in entity
        # Should be ISO format strings
        assert isinstance(entity["last_changed"], str)
        assert isinstance(entity["last_updated"], str)

    @pytest.mark.asyncio
    async def test_query_all_attributes_when_no_filter(self, mock_hass, sample_light_state):
        """Test that all attributes are returned when no filter is specified."""
        tool = HomeAssistantQueryTool(mock_hass)

        mock_hass.states.async_entity_ids.return_value = ["light.living_room"]
        mock_hass.states.get.return_value = sample_light_state

        result = await tool.execute(entity_id="light.living_room")

        entity = result["entities"][0]
        # All attributes from sample_light_state should be present
        assert "brightness" in entity["attributes"]
        assert "color_temp" in entity["attributes"]
        assert "rgb_color" in entity["attributes"]
        assert "friendly_name" in entity["attributes"]
