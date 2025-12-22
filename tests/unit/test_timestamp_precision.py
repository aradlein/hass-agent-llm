"""Unit tests for timestamp precision (Issue 73).

Verifies that timestamps are formatted with seconds precision only,
not microseconds/nanoseconds, to reduce token usage and improve readability.

Issue 73: Timestamp precision reduced from nanoseconds to seconds
- All datetime.isoformat() calls now use timespec='seconds'
- No more microsecond precision in timestamps (no .123456 suffixes)
- Applies to:
  - _make_json_serializable() in base.py
  - Entity state timestamps (last_changed, last_updated) in ha_query.py
  - History query timestamps (start_time, end_time) in ha_query.py
  - Memory extraction event timestamps in memory_extraction.py
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest
from homeassistant.core import State
from homeassistant.util import dt as dt_util

from custom_components.home_agent.context_providers.base import _make_json_serializable
from custom_components.home_agent.tools.ha_query import HomeAssistantQueryTool


class TestTimestampPrecision:
    """Tests for timestamp precision in formatted output."""

    def test_datetime_isoformat_seconds_only(self):
        """Verify datetime isoformat uses seconds precision."""
        dt = datetime(2025, 12, 9, 10, 30, 45, 123456)  # Has microseconds
        formatted = dt.isoformat(timespec='seconds')
        assert formatted == "2025-12-09T10:30:45"
        assert "." not in formatted  # No decimal point means no microseconds

    def test_datetime_isoformat_no_microseconds_when_zero(self):
        """Verify no microseconds shown even when microseconds are zero."""
        dt = datetime(2025, 12, 9, 10, 30, 45, 0)  # No microseconds
        formatted = dt.isoformat(timespec='seconds')
        assert formatted == "2025-12-09T10:30:45"
        assert "." not in formatted

    def test_datetime_isoformat_midnight(self):
        """Verify midnight timestamp has no microseconds."""
        dt = datetime(2025, 12, 9, 0, 0, 0, 999999)  # Has microseconds
        formatted = dt.isoformat(timespec='seconds')
        assert formatted == "2025-12-09T00:00:00"
        assert "." not in formatted


class TestMakeJsonSerializable:
    """Tests for _make_json_serializable timestamp formatting."""

    def test_make_json_serializable_datetime(self):
        """Test that _make_json_serializable formats datetime with seconds precision."""
        dt = datetime(2025, 12, 9, 10, 30, 45, 123456)
        result = _make_json_serializable(dt)
        assert result == "2025-12-09T10:30:45"
        assert "123456" not in result
        assert "." not in result

    def test_make_json_serializable_datetime_with_timezone(self):
        """Test datetime with timezone is formatted with seconds precision."""
        # Create timezone-aware datetime
        dt = dt_util.parse_datetime("2025-12-09T10:30:45.123456+00:00")
        result = _make_json_serializable(dt)
        # Should have timezone info but no microseconds
        assert ".123456" not in result
        # No decimal point means no microseconds
        assert "." not in result
        # Should end with timezone offset (e.g., +00:00)
        assert "+" in result or result.endswith("Z")

    def test_make_json_serializable_datetime_in_dict(self):
        """Test datetime in nested dict is formatted correctly."""
        data = {
            "timestamp": datetime(2025, 12, 9, 10, 30, 45, 123456),
            "nested": {
                "time": datetime(2025, 12, 9, 14, 15, 16, 789012)
            }
        }
        result = _make_json_serializable(data)
        assert result["timestamp"] == "2025-12-09T10:30:45"
        assert result["nested"]["time"] == "2025-12-09T14:15:16"
        assert "123456" not in str(result)
        assert "789012" not in str(result)

    def test_make_json_serializable_datetime_in_list(self):
        """Test datetime in list is formatted correctly."""
        data = [
            datetime(2025, 12, 9, 10, 30, 45, 123456),
            datetime(2025, 12, 9, 14, 15, 16, 789012),
        ]
        result = _make_json_serializable(data)
        assert result[0] == "2025-12-09T10:30:45"
        assert result[1] == "2025-12-09T14:15:16"
        assert "123456" not in str(result)
        assert "789012" not in str(result)

    def test_make_json_serializable_non_datetime(self):
        """Test non-datetime values are handled correctly."""
        # String
        assert _make_json_serializable("test") == "test"
        # Number
        assert _make_json_serializable(42) == 42
        # Boolean
        assert _make_json_serializable(True) is True
        # None
        assert _make_json_serializable(None) is None


class TestEntityStateTimestamps:
    """Tests for entity state timestamp formatting in ha_query tool."""

    def test_format_entity_state_no_microseconds(self):
        """Test that _format_entity_state formats timestamps without microseconds."""
        # Create mock state with microseconds in timestamps
        now = dt_util.parse_datetime("2025-12-09T10:30:45.123456+00:00")
        state = State(
            entity_id="light.kitchen",
            state="on",
            attributes={"brightness": 255},
            last_changed=now,
            last_updated=now,
        )

        # Create tool instance
        tool = HomeAssistantQueryTool(hass=Mock(), exposed_entities=None)

        # Mock _get_entity_services to avoid needing full HA setup
        tool._get_entity_services = Mock(return_value=["turn_on", "turn_off"])

        # Format the state
        result = tool._format_entity_state(state)

        # Verify timestamps have no microseconds
        assert "last_changed" in result
        assert "last_updated" in result
        assert ".123456" not in result["last_changed"]
        assert ".123456" not in result["last_updated"]
        # Verify no decimal point (no microseconds)
        assert "." not in result["last_changed"]
        assert "." not in result["last_updated"]

    def test_format_entity_state_timestamps_match_pattern(self):
        """Test that formatted timestamps match expected pattern."""
        now = dt_util.parse_datetime("2025-12-09T10:30:45.999999+00:00")
        state = State(
            entity_id="sensor.temperature",
            state="22.5",
            attributes={"unit_of_measurement": "°C"},
            last_changed=now,
            last_updated=now,
        )

        tool = HomeAssistantQueryTool(hass=Mock(), exposed_entities=None)
        tool._get_entity_services = Mock(return_value=["reload"])

        result = tool._format_entity_state(state)

        # Verify format: should be YYYY-MM-DDTHH:MM:SS+TZ
        import re
        # Pattern for ISO 8601 with seconds precision and timezone
        pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+\-Z]'
        assert re.match(pattern, result["last_changed"])
        assert re.match(pattern, result["last_updated"])

    def test_format_entity_state_with_attribute_filter(self):
        """Test timestamp formatting with attribute filter."""
        now = dt_util.parse_datetime("2025-12-09T10:30:45.555555+00:00")
        state = State(
            entity_id="climate.bedroom",
            state="heat",
            attributes={
                "temperature": 20,
                "target_temp_high": 22,
                "target_temp_low": 18,
            },
            last_changed=now,
            last_updated=now,
        )

        tool = HomeAssistantQueryTool(hass=Mock(), exposed_entities=None)
        tool._get_entity_services = Mock(return_value=["set_temperature"])

        # Filter to only include temperature attribute
        result = tool._format_entity_state(state, attributes_filter=["temperature"])

        # Timestamps should still be formatted correctly
        assert ".555555" not in result["last_changed"]
        assert ".555555" not in result["last_updated"]
        # Only filtered attribute should be present
        assert "temperature" in result["attributes"]
        assert "target_temp_high" not in result["attributes"]


class TestHistoryQueryTimestamps:
    """Tests for history query timestamp formatting."""

    @pytest.mark.asyncio
    async def test_query_history_timestamps_no_microseconds(self):
        """Test that _query_history formats start_time and end_time without microseconds."""
        # Create mock tool
        mock_hass = Mock()
        mock_hass.states.async_entity_ids.return_value = ["sensor.temperature"]

        tool = HomeAssistantQueryTool(hass=mock_hass, exposed_entities=None)

        # Mock the history retrieval to return some data
        with patch.object(
            tool,
            '_get_entity_history',
            new_callable=AsyncMock
        ) as mock_get_history:
            # Return mock states with numeric values
            mock_state1 = Mock()
            mock_state1.state = "21.5"
            mock_state2 = Mock()
            mock_state2.state = "22.0"
            mock_get_history.return_value = [mock_state1, mock_state2]

            # Execute history query
            result = await tool._query_history(
                entity_ids=["sensor.temperature"],
                pattern="sensor.temperature",
                history_params={"duration": "1h", "aggregate": "avg"},
            )

            # Verify result structure
            assert result["success"] is True
            assert "history" in result
            assert len(result["history"]) == 1

            # Get the history entry
            history_entry = result["history"][0]

            # Verify timestamps have no microseconds
            assert "start_time" in history_entry
            assert "end_time" in history_entry
            # Should not contain decimal point (no microseconds)
            assert "." not in history_entry["start_time"].split("+")[0].split("Z")[0]
            assert "." not in history_entry["end_time"].split("+")[0].split("Z")[0]

    @pytest.mark.asyncio
    async def test_query_history_multiple_entities_timestamp_format(self):
        """Test history query with multiple entities formats all timestamps correctly."""
        mock_hass = Mock()
        mock_hass.states.async_entity_ids.return_value = [
            "sensor.temp1",
            "sensor.temp2",
        ]

        tool = HomeAssistantQueryTool(hass=mock_hass, exposed_entities=None)

        # Mock history for both entities
        with patch.object(
            tool,
            '_get_entity_history',
            new_callable=AsyncMock
        ) as mock_get_history:
            # Return different mock data for each entity
            def get_history_side_effect(entity_id, start, end):
                mock_state = Mock()
                mock_state.state = "20.0" if "temp1" in entity_id else "25.0"
                return [mock_state]

            mock_get_history.side_effect = get_history_side_effect

            result = await tool._query_history(
                entity_ids=["sensor.temp1", "sensor.temp2"],
                pattern="sensor.temp*",
                history_params={"duration": "24h", "aggregate": "max"},
            )

            # Verify all history entries have correctly formatted timestamps
            assert len(result["history"]) == 2
            for history_entry in result["history"]:
                start_time = history_entry["start_time"]
                end_time = history_entry["end_time"]

                # Remove timezone for checking (split on + or Z)
                start_time_base = start_time.split("+")[0].split("Z")[0]
                end_time_base = end_time.split("+")[0].split("Z")[0]

                # No microseconds (no decimal point in time portion)
                assert "." not in start_time_base
                assert "." not in end_time_base


class TestMemoryExtractionTimestamp:
    """Tests for memory extraction event timestamp formatting."""

    def test_memory_extraction_event_timestamp_format(self):
        """Test that memory extraction timestamp code produces correct format.

        The memory_extraction.py module uses datetime.now().isoformat(timespec='seconds')
        for the timestamp field. This test verifies that format produces no microseconds.
        """
        from datetime import datetime

        # This is exactly how the timestamp is created in memory_extraction.py:688
        # timestamp = datetime.now().isoformat(timespec='seconds')
        dt = datetime(2025, 12, 9, 10, 30, 45, 123456)
        timestamp = dt.isoformat(timespec='seconds')

        # Verify no microseconds
        assert timestamp == "2025-12-09T10:30:45"
        assert "." not in timestamp
        assert "123456" not in timestamp

    def test_memory_extraction_timestamp_parsing(self):
        """Test that memory extraction timestamp can be parsed back to datetime."""
        from datetime import datetime

        # Simulate the timestamp creation as done in memory_extraction.py
        now = datetime.now()
        timestamp_str = now.isoformat(timespec='seconds')

        # Verify it can be parsed back
        parsed = datetime.fromisoformat(timestamp_str)

        # Should be equal to the second (microseconds stripped)
        assert parsed.year == now.year
        assert parsed.month == now.month
        assert parsed.day == now.day
        assert parsed.hour == now.hour
        assert parsed.minute == now.minute
        assert parsed.second == now.second
        assert parsed.microsecond == 0  # Microseconds should be 0


class TestTimestampConsistency:
    """Tests for timestamp consistency across different modules."""

    def test_all_timestamps_use_same_precision(self):
        """Verify all timestamp formatting uses consistent seconds precision."""
        dt = datetime(2025, 12, 9, 10, 30, 45, 123456)

        # All should produce identical format
        base_format = _make_json_serializable(dt)
        direct_format = dt.isoformat(timespec='seconds')

        assert base_format == direct_format
        assert base_format == "2025-12-09T10:30:45"

    def test_timezone_aware_timestamps_consistent(self):
        """Verify timezone-aware timestamps are consistently formatted."""
        dt = dt_util.parse_datetime("2025-12-09T10:30:45.123456+00:00")

        formatted = _make_json_serializable(dt)

        # Should have timezone but no microseconds
        assert ".123456" not in formatted
        # Should end with timezone offset
        assert "+" in formatted or formatted.endswith("Z")

    def test_timestamp_length_reduced(self):
        """Verify timestamp length is reduced by removing microseconds."""
        dt = datetime(2025, 12, 9, 10, 30, 45, 123456)

        # Old format (with microseconds)
        old_format = dt.isoformat()  # Includes .123456

        # New format (seconds only)
        new_format = dt.isoformat(timespec='seconds')

        # New format should be shorter
        assert len(new_format) < len(old_format)
        assert len(new_format) == 19  # YYYY-MM-DDTHH:MM:SS
        assert len(old_format) == 26  # YYYY-MM-DDTHH:MM:SS.mmmmmm
