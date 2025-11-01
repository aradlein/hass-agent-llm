"""Home Agent - Intelligent conversation agent for Home Assistant.

This custom component provides advanced conversational AI capabilities with
tool calling, context injection, and conversation history management.
"""

from __future__ import annotations

import logging

from homeassistant.components import conversation as ha_conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers.typing import ConfigType

from .agent import HomeAgent
from .const import (
    CONF_CONTEXT_MODE,
    CONF_MEMORY_ENABLED,
    CONF_TOOLS_CUSTOM,
    CONTEXT_MODE_VECTOR_DB,
    DEFAULT_MEMORY_ENABLED,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = []  # No additional platforms needed for conversation agent


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the Home Agent component from YAML configuration.

    Args:
        hass: Home Assistant instance
        config: Configuration dictionary

    Returns:
        True if setup was successful
    """
    # Store YAML config for later use (especially custom tools)
    hass.data.setdefault(DOMAIN, {})
    if DOMAIN in config:
        hass.data[DOMAIN]["yaml_config"] = config[DOMAIN]
        _LOGGER.info("Loaded Home Agent YAML configuration")

    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Home Agent from a config entry.

    Args:
        hass: Home Assistant instance
        entry: Config entry instance

    Returns:
        True if setup was successful
    """
    _LOGGER.info("Setting up Home Agent config entry: %s", entry.entry_id)

    # Merge config data
    config = dict(entry.data) | dict(entry.options)

    # Also merge YAML config for custom tools (if present)
    # This allows users to define custom tools in configuration.yaml
    if "yaml_config" in hass.data.get(DOMAIN, {}):
        yaml_config = hass.data[DOMAIN]["yaml_config"]
        if CONF_TOOLS_CUSTOM in yaml_config:
            config[CONF_TOOLS_CUSTOM] = yaml_config[CONF_TOOLS_CUSTOM]
            _LOGGER.info("Loaded %d custom tool(s) from YAML configuration",
                        len(yaml_config[CONF_TOOLS_CUSTOM]))

    # Create Home Agent instance
    agent = HomeAgent(hass, config)

    # Store agent instance
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = {"agent": agent}

    # Set up vector DB manager if using vector DB mode
    context_mode = config.get(CONF_CONTEXT_MODE)
    vector_manager = None
    if context_mode == CONTEXT_MODE_VECTOR_DB:
        try:
            from .vector_db_manager import VectorDBManager

            vector_manager = VectorDBManager(hass, config)
            await vector_manager.async_setup()
            hass.data[DOMAIN][entry.entry_id]["vector_manager"] = vector_manager
            _LOGGER.info("Vector DB Manager enabled for this entry")
        except Exception as err:
            _LOGGER.error("Failed to set up Vector DB Manager: %s", err)
            # Continue setup without vector DB

    # Set up memory manager if enabled
    memory_enabled = config.get(CONF_MEMORY_ENABLED, DEFAULT_MEMORY_ENABLED)
    if memory_enabled:
        try:
            from .memory_manager import MemoryManager

            memory_manager = MemoryManager(
                hass=hass,
                vector_db_manager=vector_manager,
                config=config,
            )
            await memory_manager.async_initialize()
            hass.data[DOMAIN][entry.entry_id]["memory_manager"] = memory_manager
            _LOGGER.info("Memory Manager enabled for this entry")
        except Exception as err:
            _LOGGER.error("Failed to set up Memory Manager: %s", err)
            # Continue setup without memory manager

    # Register as a conversation agent
    ha_conversation.async_set_agent(hass, entry, agent)

    # Register services
    await async_setup_services(hass, entry.entry_id)

    # Register update listener to reload on config changes
    entry.async_on_unload(entry.add_update_listener(async_reload_entry))

    _LOGGER.info("Home Agent setup complete")
    return True


async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload the config entry when it's updated.

    Args:
        hass: Home Assistant instance
        entry: Config entry that was updated
    """
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry.

    Args:
        hass: Home Assistant instance
        entry: Config entry instance

    Returns:
        True if unload was successful
    """
    _LOGGER.info("Unloading Home Agent config entry: %s", entry.entry_id)

    # Unregister conversation agent
    ha_conversation.async_unset_agent(hass, entry)

    # Clean up agent, memory manager, and vector DB manager
    if entry.entry_id in hass.data[DOMAIN]:
        entry_data = hass.data[DOMAIN][entry.entry_id]

        # Shut down memory manager if it exists
        if "memory_manager" in entry_data:
            await entry_data["memory_manager"].async_shutdown()

        # Shut down vector DB manager if it exists
        if "vector_manager" in entry_data:
            await entry_data["vector_manager"].async_shutdown()

        # Clean up agent
        agent: HomeAgent = entry_data["agent"]
        await agent.close()

        del hass.data[DOMAIN][entry.entry_id]

    # Remove services if this was the last entry
    if not hass.data[DOMAIN]:
        await async_remove_services(hass)

    return True


async def async_setup_services(
    hass: HomeAssistant,
    entry_id: str,
) -> None:
    """Register Home Agent services.

    Args:
        hass: Home Assistant instance
        entry_id: Config entry ID
    """

    def _get_entry_data(target_entry_id: str) -> dict:
        """Get entry data, defaulting to provided entry_id."""
        if target_entry_id in hass.data[DOMAIN]:
            return hass.data[DOMAIN][target_entry_id]
        return hass.data[DOMAIN].get(entry_id, {})

    async def handle_process(call: ServiceCall) -> None:
        """Handle the process service call.

        Processes a user message through the agent and returns the response.
        """
        text = call.data.get("text", "")
        conversation_id = call.data.get("conversation_id")
        user_id = call.data.get("user_id")
        target_entry_id = call.data.get("entry_id", entry_id)

        # Get the right agent instance
        entry_data = _get_entry_data(target_entry_id)
        target_agent = entry_data.get("agent")

        try:
            response = await target_agent.process_message(
                text=text,
                conversation_id=conversation_id,
                user_id=user_id,
            )

            _LOGGER.info("Processed message successfully")

            # Return response (Home Assistant will handle this)
            return {
                "response": response,
                "conversation_id": conversation_id,
            }

        except Exception as err:
            _LOGGER.error("Failed to process message: %s", err)
            raise

    async def handle_clear_history(call: ServiceCall) -> None:
        """Handle the clear_history service call.

        Clears conversation history for a specific conversation or all conversations.
        """
        conversation_id = call.data.get("conversation_id")
        target_entry_id = call.data.get("entry_id", entry_id)

        # Get the right agent instance
        entry_data = _get_entry_data(target_entry_id)
        target_agent = entry_data.get("agent")

        await target_agent.clear_history(conversation_id)

        _LOGGER.info(
            "Cleared history for %s",
            conversation_id if conversation_id else "all conversations",
        )

    async def handle_reload_context(call: ServiceCall) -> None:
        """Handle the reload_context service call.

        Reloads entity context (useful after entity changes).
        """
        target_entry_id = call.data.get("entry_id", entry_id)

        # Get the right agent instance
        entry_data = _get_entry_data(target_entry_id)
        target_agent = entry_data.get("agent")

        await target_agent.reload_context()

        _LOGGER.info("Reloaded context")

    async def handle_execute_tool(call: ServiceCall) -> None:
        """Handle the execute_tool service call (debug/testing).

        Manually executes a tool for testing purposes.
        """
        tool_name = call.data.get("tool_name", "")
        parameters = call.data.get("parameters", {})
        target_entry_id = call.data.get("entry_id", entry_id)

        # Get the right agent instance
        entry_data = _get_entry_data(target_entry_id)
        target_agent = entry_data.get("agent")

        try:
            result = await target_agent.execute_tool_debug(tool_name, parameters)

            _LOGGER.info("Executed tool %s successfully", tool_name)

            return {
                "tool_name": tool_name,
                "result": result,
            }

        except Exception as err:
            _LOGGER.error("Failed to execute tool %s: %s", tool_name, err)
            raise

    async def handle_reindex_entities(call: ServiceCall) -> None:
        """Handle the reindex_entities service call.

        Forces a full reindex of all entities into the vector database.
        """
        target_entry_id = call.data.get("entry_id", entry_id)

        # Get vector DB manager
        entry_data = _get_entry_data(target_entry_id)
        vector_manager = entry_data.get("vector_manager")

        if not vector_manager:
            _LOGGER.error("Vector DB Manager not enabled for this entry")
            return {"error": "Vector DB Manager not enabled"}

        try:
            stats = await vector_manager.async_reindex_all_entities()
            _LOGGER.info("Reindex complete: %s", stats)
            return stats

        except Exception as err:
            _LOGGER.error("Failed to reindex entities: %s", err)
            raise

    async def handle_index_entity(call: ServiceCall) -> None:
        """Handle the index_entity service call.

        Indexes a specific entity into the vector database.
        """
        entity_id = call.data.get("entity_id")
        target_entry_id = call.data.get("entry_id", entry_id)

        if not entity_id:
            _LOGGER.error("entity_id is required")
            return {"error": "entity_id is required"}

        # Get vector DB manager
        entry_data = _get_entry_data(target_entry_id)
        vector_manager = entry_data.get("vector_manager")

        if not vector_manager:
            _LOGGER.error("Vector DB Manager not enabled for this entry")
            return {"error": "Vector DB Manager not enabled"}

        try:
            await vector_manager.async_index_entity(entity_id)
            _LOGGER.info("Indexed entity: %s", entity_id)
            return {"entity_id": entity_id, "status": "indexed"}

        except Exception as err:
            _LOGGER.error("Failed to index entity %s: %s", entity_id, err)
            raise

    # Register services (only once for all instances)
    if not hass.services.has_service(DOMAIN, "process"):
        hass.services.async_register(DOMAIN, "process", handle_process)
        _LOGGER.debug("Registered service: process")

    if not hass.services.has_service(DOMAIN, "clear_history"):
        hass.services.async_register(DOMAIN, "clear_history", handle_clear_history)
        _LOGGER.debug("Registered service: clear_history")

    if not hass.services.has_service(DOMAIN, "reload_context"):
        hass.services.async_register(DOMAIN, "reload_context", handle_reload_context)
        _LOGGER.debug("Registered service: reload_context")

    if not hass.services.has_service(DOMAIN, "execute_tool"):
        hass.services.async_register(DOMAIN, "execute_tool", handle_execute_tool)
        _LOGGER.debug("Registered service: execute_tool")

    if not hass.services.has_service(DOMAIN, "reindex_entities"):
        hass.services.async_register(DOMAIN, "reindex_entities", handle_reindex_entities)
        _LOGGER.debug("Registered service: reindex_entities")

    if not hass.services.has_service(DOMAIN, "index_entity"):
        hass.services.async_register(DOMAIN, "index_entity", handle_index_entity)
        _LOGGER.debug("Registered service: index_entity")


async def async_remove_services(hass: HomeAssistant) -> None:
    """Remove Home Agent services.

    Args:
        hass: Home Assistant instance
    """
    services = [
        "process",
        "clear_history",
        "reload_context",
        "execute_tool",
        "reindex_entities",
        "index_entity",
    ]

    for service in services:
        if hass.services.has_service(DOMAIN, service):
            hass.services.async_remove(DOMAIN, service)
            _LOGGER.debug("Removed service: %s", service)
