"""Home Assistant query tool for the Home Agent integration.

This module provides the HomeAssistantQueryTool for querying entity states,
attributes, and historical data from Home Assistant.
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import re
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from homeassistant.core import HomeAssistant, State
from homeassistant.util import dt as dt_util

# Conditional imports for recorder (may not be available in test environments)
try:
    from homeassistant.components import recorder
    from homeassistant.components.recorder import history
    from homeassistant.components.recorder.util import async_migration_in_progress

    RECORDER_AVAILABLE = True
except ImportError:
    RECORDER_AVAILABLE = False

from ..const import (
    HISTORY_AGGREGATE_AVG,
    HISTORY_AGGREGATE_COUNT,
    HISTORY_AGGREGATE_MAX,
    HISTORY_AGGREGATE_MIN,
    HISTORY_AGGREGATE_SUM,
    TOOL_HA_QUERY,
)
from ..exceptions import PermissionDenied, ToolExecutionError, ValidationError
from .registry import BaseTool

if TYPE_CHECKING:
    from collections.abc import Sequence

_LOGGER = logging.getLogger(__name__)


class HomeAssistantQueryTool(BaseTool):
    """Tool for querying Home Assistant entity states and history.

    This tool allows the LLM to retrieve current state information and
    historical data from Home Assistant entities. It supports wildcard
    matching for discovering entities and optional history queries with
    aggregation.

    Features:
        - Query current entity states
        - Wildcard support (e.g., "light.*" for all lights)
        - Attribute filtering
        - Historical data queries
        - Data aggregation (avg, min, max, sum, count)

    Example tool calls:
        # Get current temperature
        {
            "entity_id": "sensor.living_room_temperature"
        }

        # Get all lights
        {
            "entity_id": "light.*"
        }

        # Get specific attributes
        {
            "entity_id": "climate.thermostat",
            "attributes": ["temperature", "hvac_mode"]
        }

        # Get average temperature over 24 hours
        {
            "entity_id": "sensor.temperature",
            "history": {
                "duration": "24h",
                "aggregate": "avg"
            }
        }
    """

    def __init__(
        self,
        hass: HomeAssistant,
        exposed_entities: set[str] | None = None,
    ) -> None:
        """Initialize the Home Assistant query tool.

        Args:
            hass: Home Assistant instance
            exposed_entities: Optional set of entity IDs that are exposed
                for querying. If None, all entities are accessible.
        """
        super().__init__(hass)
        self._exposed_entities = exposed_entities

    @property
    def name(self) -> str:
        """Return the tool name."""
        return TOOL_HA_QUERY

    @property
    def description(self) -> str:
        """Return the tool description."""
        return (
            "Get current state and attributes of Home Assistant entities. "
            "Use this to check if lights are on, get sensor values, check door "
            "lock status, or retrieve any entity state information. "
            "Supports wildcards (e.g., 'light.*' for all lights) and historical "
            "data queries. Always use this before ha_control to check current state."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        """Return the tool parameter schema."""
        return {
            "type": "object",
            "properties": {
                "entity_id": {
                    "type": "string",
                    "description": (
                        "The entity ID to query in the format 'domain.entity_name'. "
                        "Supports wildcards: 'light.*' for all lights, '*.living_room' "
                        "for all entities in living room, or specific ID like "
                        "'sensor.temperature'. Use wildcards to discover entities "
                        "when exact ID is unknown."
                    ),
                },
                "attributes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Specific attributes to retrieve. If not specified, returns "
                        "all attributes. Common attributes: 'state', 'friendly_name', "
                        "'temperature', 'brightness', 'battery_level'. Use this to "
                        "reduce response size when only specific data is needed."
                    ),
                },
                "history": {
                    "type": "object",
                    "description": (
                        "Optional: retrieve and aggregate historical data. "
                        "Useful for trend analysis, averages, min/max values over time."
                    ),
                    "properties": {
                        "duration": {
                            "type": "string",
                            "description": (
                                "Time range to query. Format: number + unit. "
                                "Examples: '1h' (1 hour), '24h' (24 hours), '7d' (7 days), "
                                "'30m' (30 minutes). Maximum: 30d (30 days)."
                            ),
                        },
                        "aggregate": {
                            "type": "string",
                            "description": (
                                "How to aggregate historical data. "
                                "'avg': average value, 'min': minimum value, "
                                "'max': maximum value, 'sum': sum of values, "
                                "'count': number of state changes."
                            ),
                            "enum": [
                                HISTORY_AGGREGATE_AVG,
                                HISTORY_AGGREGATE_MIN,
                                HISTORY_AGGREGATE_MAX,
                                HISTORY_AGGREGATE_SUM,
                                HISTORY_AGGREGATE_COUNT,
                            ],
                        },
                    },
                    "required": ["duration"],
                },
            },
            "required": ["entity_id"],
        }

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Execute a query for Home Assistant entity state(s).

        Args:
            entity_id: Entity ID to query (supports wildcards)
            attributes: Optional list of specific attributes to return
            history: Optional dict with duration and aggregation for historical data

        Returns:
            Dict containing:
                - success: bool indicating if execution succeeded
                - entity_id: The queried entity ID (or pattern)
                - entities: List of entity data dicts
                - count: Number of entities returned
                - message: Human-readable result message

        Raises:
            ValidationError: If parameters are invalid
            PermissionDenied: If entity is not accessible
            ToolExecutionError: If query fails
        """
        entity_id_pattern = kwargs.get("entity_id")
        attributes_filter = kwargs.get("attributes")
        history_params = kwargs.get("history")

        # Validate required parameters
        if not entity_id_pattern:
            raise ValidationError("Parameter 'entity_id' is required")

        # Validate entity ID pattern
        if not self._is_valid_entity_pattern(entity_id_pattern):
            raise ValidationError(
                f"Invalid entity_id pattern: '{entity_id_pattern}'. "
                f"Expected format: 'domain.entity_name' or with wildcards like 'light.*'"
            )

        try:
            # Find matching entities
            matching_entities = self._find_matching_entities(entity_id_pattern)

            if not matching_entities:
                return {
                    "success": True,
                    "entity_id": entity_id_pattern,
                    "entities": [],
                    "count": 0,
                    "message": f"No entities found matching pattern: {entity_id_pattern}",
                }

            # Check if we're querying history
            if history_params:
                return await self._query_history(
                    matching_entities,
                    entity_id_pattern,
                    history_params,
                )

            # Query current states
            entity_data = []
            for entity_id in matching_entities:
                state = self.hass.states.get(entity_id)
                if state:
                    entity_info = self._format_entity_state(
                        state,
                        attributes_filter,
                    )
                    entity_data.append(entity_info)

            result = {
                "success": True,
                "entity_id": entity_id_pattern,
                "entities": entity_data,
                "count": len(entity_data),
                "message": self._build_success_message(entity_id_pattern, len(entity_data)),
            }

            _LOGGER.info(
                "Successfully queried %d entities matching pattern: %s",
                len(entity_data),
                entity_id_pattern,
            )

            return result

        except (ValidationError, PermissionDenied):
            # Re-raise validation and permission errors as-is
            raise
        except Exception as error:
            _LOGGER.error(
                "Failed to query entities matching %s: %s",
                entity_id_pattern,
                error,
                exc_info=True,
            )
            raise ToolExecutionError(
                f"Failed to query entities matching {entity_id_pattern}: {error}"
            ) from error

    def _is_valid_entity_pattern(self, pattern: str) -> bool:
        """Validate entity ID pattern.

        Args:
            pattern: Entity ID pattern to validate

        Returns:
            True if pattern is valid
        """
        # Must have at least a dot
        if "." not in pattern:
            return False

        # Split on dot
        parts = pattern.split(".")

        # Must have exactly 2 parts (domain.entity_name)
        if len(parts) != 2:
            return False

        # Each part can be wildcard or alphanumeric with underscores
        for part in parts:
            if part != "*" and not re.match(r"^[a-z0-9_*]+$", part):
                return False

        return True

    def _find_matching_entities(self, pattern: str) -> list[str]:
        """Find all entities matching the pattern.

        Args:
            pattern: Entity ID pattern (may include wildcards)

        Returns:
            List of matching entity IDs

        Raises:
            PermissionDenied: If any matched entity is not accessible
        """
        # Get all entity IDs from state machine
        all_entities = self.hass.states.async_entity_ids()

        # Filter by pattern
        if "*" in pattern:
            # Use fnmatch for wildcard matching
            matching = [e for e in all_entities if fnmatch.fnmatch(e, pattern)]
        else:
            # Exact match
            matching = [pattern] if pattern in all_entities else []

        # Validate access to matched entities
        for entity_id in matching:
            self._validate_entity_access(entity_id)

        return matching

    def _validate_entity_access(self, entity_id: str) -> None:
        """Validate that the entity is accessible.

        Args:
            entity_id: Entity ID to validate

        Raises:
            PermissionDenied: If entity is not accessible
        """
        # If no exposed entities set is provided, allow all access
        if self._exposed_entities is None:
            return

        if entity_id not in self._exposed_entities:
            _LOGGER.warning(
                "Attempted access to unexposed entity: %s",
                entity_id,
            )
            raise PermissionDenied(
                f"Entity '{entity_id}' is not accessible. "
                f"Ensure it is exposed in the integration configuration or "
                f"voice assistant settings."
            )

    def _format_entity_state(
        self,
        state: State,
        attributes_filter: list[str] | None = None,
    ) -> dict[str, Any]:
        """Format entity state for LLM consumption.

        Args:
            state: Entity state object
            attributes_filter: Optional list of attributes to include

        Returns:
            Dict containing entity information
        """
        entity_data: dict[str, Any] = {
            "entity_id": state.entity_id,
            "state": state.state,
            "last_changed": state.last_changed.isoformat(),
            "last_updated": state.last_updated.isoformat(),
        }

        # Add attributes
        if attributes_filter:
            # Only include requested attributes
            entity_data["attributes"] = {
                attr: state.attributes.get(attr)
                for attr in attributes_filter
                if attr in state.attributes
            }
        else:
            # Include all attributes
            entity_data["attributes"] = dict(state.attributes)

        return entity_data

    async def _query_history(
        self,
        entity_ids: list[str],
        pattern: str,
        history_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Query historical data for entities.

        Args:
            entity_ids: List of entity IDs to query
            pattern: Original entity ID pattern
            history_params: History parameters (duration, aggregate)

        Returns:
            Dict containing historical query results

        Raises:
            ValidationError: If history parameters are invalid
            ToolExecutionError: If history query fails
        """
        # Validate duration
        duration_str = history_params.get("duration")
        if not duration_str:
            raise ValidationError("History parameter 'duration' is required")

        duration = self._parse_duration(duration_str)
        if duration is None:
            raise ValidationError(
                f"Invalid duration format: '{duration_str}'. "
                f"Use format like '1h', '24h', '7d', '30m'"
            )

        # Validate max duration (30 days)
        max_duration = timedelta(days=30)
        if duration > max_duration:
            raise ValidationError(f"Duration {duration_str} exceeds maximum of 30 days")

        # Get aggregation type
        aggregate = history_params.get("aggregate", HISTORY_AGGREGATE_AVG)

        # Calculate time range
        end_time = dt_util.now()
        start_time = end_time - duration

        # Query history for each entity
        entity_history_data = []

        for entity_id in entity_ids:
            try:
                # Get history from recorder
                entity_history = await self._get_entity_history(
                    entity_id,
                    start_time,
                    end_time,
                )

                if entity_history:
                    # Apply aggregation
                    aggregated_value = self._aggregate_history(
                        entity_history,
                        aggregate,
                    )

                    entity_history_data.append(
                        {
                            "entity_id": entity_id,
                            "aggregate": aggregate,
                            "value": aggregated_value,
                            "data_points": len(entity_history),
                            "start_time": start_time.isoformat(),
                            "end_time": end_time.isoformat(),
                        }
                    )

            except ToolExecutionError:
                # Re-raise tool execution errors (e.g., recorder not available)
                raise
            except Exception as error:
                _LOGGER.warning(
                    "Failed to get history for %s: %s",
                    entity_id,
                    error,
                )
                # Continue with other entities

        return {
            "success": True,
            "entity_id": pattern,
            "history": entity_history_data,
            "count": len(entity_history_data),
            "duration": duration_str,
            "aggregate": aggregate,
            "message": f"Retrieved {aggregate} values for {len(entity_history_data)} "
            f"entities over {duration_str}",
        }

    async def _get_entity_history(
        self,
        entity_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> Sequence[State]:
        """Get history for a single entity.

        Args:
            entity_id: Entity ID to query
            start_time: Start of time range
            end_time: End of time range

        Returns:
            List of state objects

        Raises:
            ToolExecutionError: If recorder is not available
        """
        # Check if recorder is available
        if not RECORDER_AVAILABLE:
            raise ToolExecutionError(
                "Recorder component is not available. "
                "Historical queries require the recorder integration."
            )

        # Check if migration is in progress
        if async_migration_in_progress(self.hass):
            raise ToolExecutionError(
                "Recorder migration is in progress. "
                "Historical queries are not available during migration."
            )

        # Check if recorder is ready
        instance = recorder.get_instance(self.hass)
        if instance is None:
            raise ToolExecutionError(
                "Recorder component is not available. "
                "Historical queries require the recorder integration."
            )

        # Wait for recorder to be ready (with timeout)
        try:
            await asyncio.wait_for(instance.async_recorder_ready.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            raise ToolExecutionError(
                "Recorder is not ready. "
                "Historical queries require the recorder integration to be initialized."
            )

        # Get history from recorder
        entity_history = await self.hass.async_add_executor_job(
            history.state_changes_during_period,
            self.hass,
            start_time,
            end_time,
            entity_id,
        )

        return list(entity_history.get(entity_id, []))

    def _aggregate_history(
        self,
        states: Sequence[State],
        aggregate: str,
    ) -> float | int | None:
        """Aggregate historical state data.

        Args:
            states: List of state objects
            aggregate: Aggregation type (avg, min, max, sum, count)

        Returns:
            Aggregated value, or None if no numeric data available
        """
        if not states:
            return None

        if aggregate == HISTORY_AGGREGATE_COUNT:
            return len(states)

        # Extract numeric values
        numeric_values = []
        for state in states:
            try:
                # Try to convert state to float
                value = float(state.state)
                numeric_values.append(value)
            except (ValueError, TypeError):
                # Skip non-numeric states
                continue

        if not numeric_values:
            return None

        # Apply aggregation
        if aggregate == HISTORY_AGGREGATE_AVG:
            return sum(numeric_values) / len(numeric_values)
        elif aggregate == HISTORY_AGGREGATE_MIN:
            return min(numeric_values)
        elif aggregate == HISTORY_AGGREGATE_MAX:
            return max(numeric_values)
        elif aggregate == HISTORY_AGGREGATE_SUM:
            return sum(numeric_values)
        else:
            _LOGGER.warning("Unknown aggregate type: %s", aggregate)
            return None

    def _parse_duration(self, duration_str: str) -> timedelta | None:
        """Parse duration string to timedelta.

        Args:
            duration_str: Duration string (e.g., '1h', '24h', '7d')

        Returns:
            timedelta object, or None if invalid format
        """
        # Match pattern: number + unit
        match = re.match(r"^(\d+)([smhd])$", duration_str.lower())
        if not match:
            return None

        value = int(match.group(1))
        unit = match.group(2)

        # Convert to timedelta
        if unit == "s":
            return timedelta(seconds=value)
        elif unit == "m":
            return timedelta(minutes=value)
        elif unit == "h":
            return timedelta(hours=value)
        elif unit == "d":
            return timedelta(days=value)

        return None

    def _build_success_message(self, pattern: str, count: int) -> str:
        """Build a human-readable success message.

        Args:
            pattern: Entity ID pattern that was queried
            count: Number of entities found

        Returns:
            Human-readable success message
        """
        if count == 0:
            return f"No entities found matching pattern: {pattern}"
        elif count == 1:
            return f"Found 1 entity matching pattern: {pattern}"
        else:
            return f"Found {count} entities matching pattern: {pattern}"
