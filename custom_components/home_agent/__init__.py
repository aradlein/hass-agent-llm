"""Home Agent - Intelligent conversation agent for Home Assistant.

This custom component provides advanced conversational AI capabilities with
tool calling, context injection, and conversation history management.
"""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.components import conversation as ha_conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers.typing import ConfigType

from .agent import HomeAgent
from .const import DOMAIN

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
    # Store empty dict for this domain
    hass.data.setdefault(DOMAIN, {})

    _LOGGER.info("Home Agent component setup (YAML config not used, use UI)")
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

    # Create Home Agent instance
    agent = HomeAgent(hass, dict(entry.data) | dict(entry.options))

    # Store agent instance
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = agent

    # Register as a conversation agent
    ha_conversation.async_set_agent(hass, entry, agent)

    # Register services
    await async_setup_services(hass, agent, entry.entry_id)

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

    # Clean up agent
    if entry.entry_id in hass.data[DOMAIN]:
        agent: HomeAgent = hass.data[DOMAIN][entry.entry_id]
        await agent.close()
        del hass.data[DOMAIN][entry.entry_id]

    # Remove services if this was the last entry
    if not hass.data[DOMAIN]:
        await async_remove_services(hass)

    return True


async def async_setup_services(
    hass: HomeAssistant,
    agent: HomeAgent,
    entry_id: str,
) -> None:
    """Register Home Agent services.

    Args:
        hass: Home Assistant instance
        agent: HomeAgent instance
        entry_id: Config entry ID
    """

    async def handle_process(call: ServiceCall) -> None:
        """Handle the process service call.

        Processes a user message through the agent and returns the response.
        """
        text = call.data.get("text", "")
        conversation_id = call.data.get("conversation_id")
        user_id = call.data.get("user_id")
        target_entry_id = call.data.get("entry_id", entry_id)

        # Get the right agent instance
        target_agent = hass.data[DOMAIN].get(target_entry_id, agent)

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
        target_agent = hass.data[DOMAIN].get(target_entry_id, agent)

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
        target_agent = hass.data[DOMAIN].get(target_entry_id, agent)

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
        target_agent = hass.data[DOMAIN].get(target_entry_id, agent)

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


async def async_remove_services(hass: HomeAssistant) -> None:
    """Remove Home Agent services.

    Args:
        hass: Home Assistant instance
    """
    services = ["process", "clear_history", "reload_context", "execute_tool"]

    for service in services:
        if hass.services.has_service(DOMAIN, service):
            hass.services.async_remove(DOMAIN, service)
            _LOGGER.debug("Removed service: %s", service)
