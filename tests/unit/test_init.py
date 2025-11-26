"""Unit tests for Home Agent __init__.py (setup and service handlers)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall

from custom_components.home_agent import (
    async_reload_entry,
    async_remove_services,
    async_setup,
    async_setup_entry,
    async_setup_services,
    async_unload_entry,
)
from custom_components.home_agent.const import (
    CONF_CONTEXT_MODE,
    CONF_MEMORY_ENABLED,
    CONF_TOOLS_CUSTOM,
    CONTEXT_MODE_VECTOR_DB,
    DEFAULT_MEMORY_ENABLED,
    DOMAIN,
)


@pytest.fixture
def mock_hass():
    """Create a mock Home Assistant instance."""
    hass = MagicMock(spec=HomeAssistant)
    hass.data = {}
    hass.services = MagicMock()
    hass.services.has_service = MagicMock(return_value=False)
    hass.services.async_register = MagicMock()
    hass.services.async_remove = MagicMock()
    hass.config_entries = MagicMock()
    hass.config_entries.async_reload = AsyncMock()
    # Mock config.config_dir for ConversationSessionManager storage
    hass.config = MagicMock()
    hass.config.config_dir = "/tmp/test_config"
    return hass


@pytest.fixture
def mock_config_entry():
    """Create a mock config entry."""
    entry = MagicMock(spec=ConfigEntry)
    entry.entry_id = "test_entry_123"
    entry.data = {
        "llm_base_url": "https://api.openai.com/v1",
        "llm_api_key": "test-key",
        "llm_model": "gpt-4o-mini",
    }
    entry.options = {}
    entry.add_update_listener = MagicMock(return_value=lambda: None)
    entry.async_on_unload = MagicMock()
    return entry


@pytest.fixture
def mock_agent():
    """Create a mock HomeAgent."""
    agent = MagicMock()
    agent.close = AsyncMock()
    agent.process_message = AsyncMock(return_value="Test response")
    agent.clear_history = AsyncMock()
    agent.reload_context = AsyncMock()
    agent.execute_tool_debug = AsyncMock(return_value={"result": "success"})
    return agent


@pytest.fixture
def mock_memory_manager():
    """Create a mock MemoryManager."""
    manager = MagicMock()
    manager.async_initialize = AsyncMock()
    manager.async_shutdown = AsyncMock()
    manager.list_all_memories = AsyncMock(return_value=[])
    manager.delete_memory = AsyncMock(return_value=True)
    manager.clear_all_memories = AsyncMock(return_value=0)
    manager.search_memories = AsyncMock(return_value=[])
    manager.add_memory = AsyncMock(return_value="mem_123")
    return manager


@pytest.fixture
def mock_vector_manager():
    """Create a mock VectorDBManager."""
    manager = MagicMock()
    manager.async_setup = AsyncMock()
    manager.async_shutdown = AsyncMock()
    manager.async_reindex_all_entities = AsyncMock(return_value={"indexed": 10})
    manager.async_index_entity = AsyncMock()
    return manager


class TestAsyncSetup:
    """Test async_setup function."""

    async def test_setup_without_yaml_config(self, mock_hass):
        """Test setup when no YAML config is provided."""
        config = {}
        result = await async_setup(mock_hass, config)

        assert result is True
        assert DOMAIN in mock_hass.data

    async def test_setup_with_yaml_config(self, mock_hass):
        """Test setup with YAML configuration."""
        config = {
            DOMAIN: {
                CONF_TOOLS_CUSTOM: [
                    {
                        "name": "custom_tool",
                        "description": "A custom tool",
                        "parameters": {},
                    }
                ]
            }
        }
        result = await async_setup(mock_hass, config)

        assert result is True
        assert DOMAIN in mock_hass.data
        assert "yaml_config" in mock_hass.data[DOMAIN]
        assert CONF_TOOLS_CUSTOM in mock_hass.data[DOMAIN]["yaml_config"]

    async def test_setup_stores_yaml_config(self, mock_hass):
        """Test that YAML config is stored correctly."""
        custom_tools = [{"name": "tool1"}, {"name": "tool2"}]
        config = {DOMAIN: {CONF_TOOLS_CUSTOM: custom_tools}}

        result = await async_setup(mock_hass, config)

        assert result is True
        assert mock_hass.data[DOMAIN]["yaml_config"][CONF_TOOLS_CUSTOM] == custom_tools


class TestAsyncSetupEntry:
    """Test async_setup_entry function."""

    @patch("custom_components.home_agent.ConversationSessionManager")
    @patch("custom_components.home_agent.HomeAgent")
    @patch("custom_components.home_agent.ha_conversation.async_set_agent")
    @patch("custom_components.home_agent.async_setup_services")
    async def test_setup_entry_basic(
        self,
        mock_setup_services,
        mock_set_agent,
        mock_agent_class,
        mock_session_manager_class,
        mock_hass,
        mock_config_entry,
        mock_agent,
    ):
        """Test basic config entry setup without memory or vector DB."""
        mock_agent_class.return_value = mock_agent
        mock_setup_services.return_value = AsyncMock()
        mock_session_manager = MagicMock()
        mock_session_manager.async_load = AsyncMock()
        mock_session_manager_class.return_value = mock_session_manager

        result = await async_setup_entry(mock_hass, mock_config_entry)

        assert result is True
        assert DOMAIN in mock_hass.data
        assert mock_config_entry.entry_id in mock_hass.data[DOMAIN]
        assert "agent" in mock_hass.data[DOMAIN][mock_config_entry.entry_id]
        mock_set_agent.assert_called_once_with(mock_hass, mock_config_entry, mock_agent)
        mock_setup_services.assert_called_once()

    @patch("custom_components.home_agent.ConversationSessionManager")
    @patch("custom_components.home_agent.HomeAgent")
    @patch("custom_components.home_agent.ha_conversation.async_set_agent")
    @patch("custom_components.home_agent.async_setup_services")
    async def test_setup_entry_with_yaml_custom_tools(
        self,
        mock_setup_services,
        mock_set_agent,
        mock_agent_class,
        mock_session_manager_class,
        mock_hass,
        mock_config_entry,
        mock_agent,
    ):
        """Test setup merges custom tools from YAML config."""
        mock_agent_class.return_value = mock_agent
        mock_setup_services.return_value = AsyncMock()
        mock_session_manager = MagicMock()
        mock_session_manager.async_load = AsyncMock()
        mock_session_manager_class.return_value = mock_session_manager

        # Add YAML config with custom tools
        custom_tools = [{"name": "yaml_tool"}]
        mock_hass.data[DOMAIN] = {"yaml_config": {CONF_TOOLS_CUSTOM: custom_tools}}

        result = await async_setup_entry(mock_hass, mock_config_entry)

        assert result is True
        # Verify custom tools were passed to agent
        call_args = mock_agent_class.call_args
        assert CONF_TOOLS_CUSTOM in call_args[0][1]
        assert call_args[0][1][CONF_TOOLS_CUSTOM] == custom_tools

    @patch("custom_components.home_agent.ConversationSessionManager")
    @patch("custom_components.home_agent.HomeAgent")
    @patch("custom_components.home_agent.vector_db_manager.VectorDBManager")
    @patch("custom_components.home_agent.ha_conversation.async_set_agent")
    @patch("custom_components.home_agent.async_setup_services")
    async def test_setup_entry_with_vector_db(
        self,
        mock_setup_services,
        mock_set_agent,
        mock_vector_class,
        mock_agent_class,
        mock_session_manager_class,
        mock_hass,
        mock_config_entry,
        mock_agent,
        mock_vector_manager,
    ):
        """Test setup with vector DB enabled."""
        mock_agent_class.return_value = mock_agent
        mock_vector_class.return_value = mock_vector_manager
        mock_setup_services.return_value = AsyncMock()
        mock_session_manager = MagicMock()
        mock_session_manager.async_load = AsyncMock()
        mock_session_manager_class.return_value = mock_session_manager

        # Enable vector DB mode
        mock_config_entry.data = {
            **mock_config_entry.data,
            CONF_CONTEXT_MODE: CONTEXT_MODE_VECTOR_DB,
        }

        result = await async_setup_entry(mock_hass, mock_config_entry)

        assert result is True
        assert "vector_manager" in mock_hass.data[DOMAIN][mock_config_entry.entry_id]
        mock_vector_manager.async_setup.assert_called_once()

    @patch("custom_components.home_agent.ConversationSessionManager")
    @patch("custom_components.home_agent.HomeAgent")
    @patch("custom_components.home_agent.vector_db_manager.VectorDBManager")
    @patch("custom_components.home_agent.ha_conversation.async_set_agent")
    @patch("custom_components.home_agent.async_setup_services")
    async def test_setup_entry_vector_db_error_continues(
        self,
        mock_setup_services,
        mock_set_agent,
        mock_vector_class,
        mock_agent_class,
        mock_session_manager_class,
        mock_hass,
        mock_config_entry,
        mock_agent,
    ):
        """Test that setup continues if vector DB fails."""
        mock_agent_class.return_value = mock_agent
        mock_setup_services.return_value = AsyncMock()
        mock_session_manager = MagicMock()
        mock_session_manager.async_load = AsyncMock()
        mock_session_manager_class.return_value = mock_session_manager

        # Make vector DB setup fail
        mock_vector_class.side_effect = Exception("Vector DB error")

        mock_config_entry.data = {
            **mock_config_entry.data,
            CONF_CONTEXT_MODE: CONTEXT_MODE_VECTOR_DB,
        }

        result = await async_setup_entry(mock_hass, mock_config_entry)

        # Setup should still succeed
        assert result is True
        assert "vector_manager" not in mock_hass.data[DOMAIN][mock_config_entry.entry_id]

    @patch("custom_components.home_agent.ConversationSessionManager")
    @patch("custom_components.home_agent.HomeAgent")
    @patch("custom_components.home_agent.memory_manager.MemoryManager")
    @patch("custom_components.home_agent.ha_conversation.async_set_agent")
    @patch("custom_components.home_agent.async_setup_services")
    async def test_setup_entry_with_memory_enabled(
        self,
        mock_setup_services,
        mock_set_agent,
        mock_memory_class,
        mock_agent_class,
        mock_session_manager_class,
        mock_hass,
        mock_config_entry,
        mock_agent,
        mock_memory_manager,
    ):
        """Test setup with memory enabled."""
        mock_agent_class.return_value = mock_agent
        mock_memory_class.return_value = mock_memory_manager
        mock_setup_services.return_value = AsyncMock()
        mock_session_manager = MagicMock()
        mock_session_manager.async_load = AsyncMock()
        mock_session_manager_class.return_value = mock_session_manager

        # Enable memory
        mock_config_entry.options = {CONF_MEMORY_ENABLED: True}

        result = await async_setup_entry(mock_hass, mock_config_entry)

        assert result is True
        assert "memory_manager" in mock_hass.data[DOMAIN][mock_config_entry.entry_id]
        mock_memory_manager.async_initialize.assert_called_once()

    @patch("custom_components.home_agent.ConversationSessionManager")
    @patch("custom_components.home_agent.HomeAgent")
    @patch("custom_components.home_agent.memory_manager.MemoryManager")
    @patch("custom_components.home_agent.ha_conversation.async_set_agent")
    @patch("custom_components.home_agent.async_setup_services")
    async def test_setup_entry_memory_disabled_by_default(
        self,
        mock_setup_services,
        mock_set_agent,
        mock_memory_class,
        mock_agent_class,
        mock_session_manager_class,
        mock_hass,
        mock_config_entry,
        mock_agent,
    ):
        """Test that memory is disabled when not explicitly enabled."""
        mock_agent_class.return_value = mock_agent
        mock_setup_services.return_value = AsyncMock()
        mock_session_manager = MagicMock()
        mock_session_manager.async_load = AsyncMock()
        mock_session_manager_class.return_value = mock_session_manager

        # Memory defaults to DEFAULT_MEMORY_ENABLED (should be False)
        result = await async_setup_entry(mock_hass, mock_config_entry)

        assert result is True
        # Memory manager should not be created if disabled
        if not DEFAULT_MEMORY_ENABLED:
            mock_memory_class.assert_not_called()

    @patch("custom_components.home_agent.ConversationSessionManager")
    @patch("custom_components.home_agent.HomeAgent")
    @patch("custom_components.home_agent.memory_manager.MemoryManager")
    @patch("custom_components.home_agent.ha_conversation.async_set_agent")
    @patch("custom_components.home_agent.async_setup_services")
    async def test_setup_entry_memory_error_continues(
        self,
        mock_setup_services,
        mock_set_agent,
        mock_memory_class,
        mock_agent_class,
        mock_session_manager_class,
        mock_hass,
        mock_config_entry,
        mock_agent,
    ):
        """Test that setup continues if memory manager fails."""
        mock_agent_class.return_value = mock_agent
        mock_setup_services.return_value = AsyncMock()
        mock_session_manager = MagicMock()
        mock_session_manager.async_load = AsyncMock()
        mock_session_manager_class.return_value = mock_session_manager

        # Make memory manager setup fail
        mock_memory_class.side_effect = Exception("Memory error")

        mock_config_entry.options = {CONF_MEMORY_ENABLED: True}

        result = await async_setup_entry(mock_hass, mock_config_entry)

        # Setup should still succeed
        assert result is True
        assert "memory_manager" not in mock_hass.data[DOMAIN][mock_config_entry.entry_id]

    @patch("custom_components.home_agent.ConversationSessionManager")
    @patch("custom_components.home_agent.HomeAgent")
    @patch("custom_components.home_agent.vector_db_manager.VectorDBManager")
    @patch("custom_components.home_agent.memory_manager.MemoryManager")
    @patch("custom_components.home_agent.ha_conversation.async_set_agent")
    @patch("custom_components.home_agent.async_setup_services")
    async def test_setup_entry_with_both_vector_and_memory(
        self,
        mock_setup_services,
        mock_set_agent,
        mock_memory_class,
        mock_vector_class,
        mock_agent_class,
        mock_session_manager_class,
        mock_hass,
        mock_config_entry,
        mock_agent,
        mock_memory_manager,
        mock_vector_manager,
    ):
        """Test setup with both vector DB and memory enabled."""
        mock_agent_class.return_value = mock_agent
        mock_vector_class.return_value = mock_vector_manager
        mock_memory_class.return_value = mock_memory_manager
        mock_setup_services.return_value = AsyncMock()
        mock_session_manager = MagicMock()
        mock_session_manager.async_load = AsyncMock()
        mock_session_manager_class.return_value = mock_session_manager

        mock_config_entry.data = {
            **mock_config_entry.data,
            CONF_CONTEXT_MODE: CONTEXT_MODE_VECTOR_DB,
        }
        mock_config_entry.options = {CONF_MEMORY_ENABLED: True}

        result = await async_setup_entry(mock_hass, mock_config_entry)

        assert result is True
        assert "agent" in mock_hass.data[DOMAIN][mock_config_entry.entry_id]
        assert "vector_manager" in mock_hass.data[DOMAIN][mock_config_entry.entry_id]
        assert "memory_manager" in mock_hass.data[DOMAIN][mock_config_entry.entry_id]

        # Verify memory manager was passed the vector manager
        call_args = mock_memory_class.call_args
        assert call_args[1]["vector_db_manager"] == mock_vector_manager

    @patch("custom_components.home_agent.ConversationSessionManager")
    @patch("custom_components.home_agent.HomeAgent")
    @patch("custom_components.home_agent.ha_conversation.async_set_agent")
    @patch("custom_components.home_agent.async_setup_services")
    async def test_setup_entry_registers_update_listener(
        self,
        mock_setup_services,
        mock_set_agent,
        mock_agent_class,
        mock_session_manager_class,
        mock_hass,
        mock_config_entry,
        mock_agent,
    ):
        """Test that update listener is registered."""
        mock_agent_class.return_value = mock_agent
        mock_setup_services.return_value = AsyncMock()
        mock_session_manager = MagicMock()
        mock_session_manager.async_load = AsyncMock()
        mock_session_manager_class.return_value = mock_session_manager

        result = await async_setup_entry(mock_hass, mock_config_entry)

        assert result is True
        mock_config_entry.add_update_listener.assert_called_once()
        mock_config_entry.async_on_unload.assert_called_once()

    @patch("custom_components.home_agent.ConversationSessionManager")
    @patch("custom_components.home_agent.HomeAgent")
    @patch("custom_components.home_agent.ha_conversation.async_set_agent")
    @patch("custom_components.home_agent.async_setup_services")
    async def test_setup_entry_session_persistence_enabled_default(
        self,
        mock_setup_services,
        mock_set_agent,
        mock_agent_class,
        mock_session_manager_class,
        mock_hass,
        mock_config_entry,
        mock_agent,
    ):
        """Test that when no session_persistence_enabled is specified, it defaults to enabled with 3600 second timeout."""
        mock_agent_class.return_value = mock_agent
        mock_setup_services.return_value = AsyncMock()
        mock_session_manager = MagicMock()
        mock_session_manager.async_load = AsyncMock()
        mock_session_manager_class.return_value = mock_session_manager

        # Don't set session_persistence_enabled or session_timeout, so defaults are used
        result = await async_setup_entry(mock_hass, mock_config_entry)

        assert result is True
        # Verify ConversationSessionManager was called with default timeout (3600 seconds)
        mock_session_manager_class.assert_called_once_with(mock_hass, 3600)
        mock_session_manager.async_load.assert_called_once()

    @patch("custom_components.home_agent.ConversationSessionManager")
    @patch("custom_components.home_agent.HomeAgent")
    @patch("custom_components.home_agent.ha_conversation.async_set_agent")
    @patch("custom_components.home_agent.async_setup_services")
    async def test_setup_entry_session_persistence_disabled(
        self,
        mock_setup_services,
        mock_set_agent,
        mock_agent_class,
        mock_session_manager_class,
        mock_hass,
        mock_config_entry,
        mock_agent,
    ):
        """Test that when session_persistence_enabled=False, the session_manager is initialized with timeout=0."""
        mock_agent_class.return_value = mock_agent
        mock_setup_services.return_value = AsyncMock()
        mock_session_manager = MagicMock()
        mock_session_manager.async_load = AsyncMock()
        mock_session_manager_class.return_value = mock_session_manager

        # Disable session persistence
        mock_config_entry.data = {
            **mock_config_entry.data,
            "session_persistence_enabled": False,
        }

        result = await async_setup_entry(mock_hass, mock_config_entry)

        assert result is True
        # Verify ConversationSessionManager was called with timeout=0
        mock_session_manager_class.assert_called_once_with(mock_hass, 0)
        mock_session_manager.async_load.assert_called_once()

    @patch("custom_components.home_agent.ConversationSessionManager")
    @patch("custom_components.home_agent.HomeAgent")
    @patch("custom_components.home_agent.ha_conversation.async_set_agent")
    @patch("custom_components.home_agent.async_setup_services")
    async def test_setup_entry_session_persistence_custom_timeout(
        self,
        mock_setup_services,
        mock_set_agent,
        mock_agent_class,
        mock_session_manager_class,
        mock_hass,
        mock_config_entry,
        mock_agent,
    ):
        """Test that when session_timeout is set to a custom value (in minutes), it's converted to seconds correctly."""
        mock_agent_class.return_value = mock_agent
        mock_setup_services.return_value = AsyncMock()
        mock_session_manager = MagicMock()
        mock_session_manager.async_load = AsyncMock()
        mock_session_manager_class.return_value = mock_session_manager

        # Set custom session timeout (30 minutes, which should be converted to 1800 seconds)
        mock_config_entry.data = {
            **mock_config_entry.data,
            "session_timeout": 30,  # in minutes
        }

        result = await async_setup_entry(mock_hass, mock_config_entry)

        assert result is True
        # Verify ConversationSessionManager was called with 1800 seconds (30 * 60)
        mock_session_manager_class.assert_called_once_with(mock_hass, 1800)
        mock_session_manager.async_load.assert_called_once()


class TestAsyncReloadEntry:
    """Test async_reload_entry function."""

    async def test_reload_entry(self, mock_hass, mock_config_entry):
        """Test reloading a config entry."""
        await async_reload_entry(mock_hass, mock_config_entry)

        mock_hass.config_entries.async_reload.assert_called_once_with(mock_config_entry.entry_id)


class TestAsyncUnloadEntry:
    """Test async_unload_entry function."""

    @patch("custom_components.home_agent.ha_conversation.async_unset_agent")
    async def test_unload_entry_with_agent_only(
        self, mock_unset_agent, mock_hass, mock_config_entry, mock_agent
    ):
        """Test unloading entry with only agent."""
        mock_hass.data[DOMAIN] = {mock_config_entry.entry_id: {"agent": mock_agent}}

        result = await async_unload_entry(mock_hass, mock_config_entry)

        assert result is True
        mock_unset_agent.assert_called_once_with(mock_hass, mock_config_entry)
        mock_agent.close.assert_called_once()
        assert mock_config_entry.entry_id not in mock_hass.data[DOMAIN]

    @patch("custom_components.home_agent.ha_conversation.async_unset_agent")
    async def test_unload_entry_with_memory_manager(
        self,
        mock_unset_agent,
        mock_hass,
        mock_config_entry,
        mock_agent,
        mock_memory_manager,
    ):
        """Test unloading entry with memory manager."""
        mock_hass.data[DOMAIN] = {
            mock_config_entry.entry_id: {
                "agent": mock_agent,
                "memory_manager": mock_memory_manager,
            }
        }

        result = await async_unload_entry(mock_hass, mock_config_entry)

        assert result is True
        mock_memory_manager.async_shutdown.assert_called_once()
        mock_agent.close.assert_called_once()

    @patch("custom_components.home_agent.ha_conversation.async_unset_agent")
    async def test_unload_entry_with_vector_manager(
        self,
        mock_unset_agent,
        mock_hass,
        mock_config_entry,
        mock_agent,
        mock_vector_manager,
    ):
        """Test unloading entry with vector DB manager."""
        mock_hass.data[DOMAIN] = {
            mock_config_entry.entry_id: {
                "agent": mock_agent,
                "vector_manager": mock_vector_manager,
            }
        }

        result = await async_unload_entry(mock_hass, mock_config_entry)

        assert result is True
        mock_vector_manager.async_shutdown.assert_called_once()
        mock_agent.close.assert_called_once()

    @patch("custom_components.home_agent.ha_conversation.async_unset_agent")
    async def test_unload_entry_with_all_managers(
        self,
        mock_unset_agent,
        mock_hass,
        mock_config_entry,
        mock_agent,
        mock_memory_manager,
        mock_vector_manager,
    ):
        """Test unloading entry with all managers."""
        mock_hass.data[DOMAIN] = {
            mock_config_entry.entry_id: {
                "agent": mock_agent,
                "memory_manager": mock_memory_manager,
                "vector_manager": mock_vector_manager,
            }
        }

        result = await async_unload_entry(mock_hass, mock_config_entry)

        assert result is True
        mock_memory_manager.async_shutdown.assert_called_once()
        mock_vector_manager.async_shutdown.assert_called_once()
        mock_agent.close.assert_called_once()

    @patch("custom_components.home_agent.ha_conversation.async_unset_agent")
    @patch("custom_components.home_agent.async_remove_services")
    async def test_unload_entry_removes_services_when_last(
        self,
        mock_remove_services,
        mock_unset_agent,
        mock_hass,
        mock_config_entry,
        mock_agent,
    ):
        """Test that services are removed when last entry is unloaded."""
        mock_remove_services.return_value = AsyncMock()
        mock_hass.data[DOMAIN] = {mock_config_entry.entry_id: {"agent": mock_agent}}

        result = await async_unload_entry(mock_hass, mock_config_entry)

        assert result is True
        # Services should be removed since DOMAIN data is now empty
        mock_remove_services.assert_called_once_with(mock_hass)

    @patch("custom_components.home_agent.ha_conversation.async_unset_agent")
    @patch("custom_components.home_agent.async_remove_services")
    async def test_unload_entry_keeps_services_when_other_entries_exist(
        self,
        mock_remove_services,
        mock_unset_agent,
        mock_hass,
        mock_config_entry,
        mock_agent,
    ):
        """Test that services are kept when other entries exist."""
        mock_remove_services.return_value = AsyncMock()

        # Add another entry
        mock_hass.data[DOMAIN] = {
            mock_config_entry.entry_id: {"agent": mock_agent},
            "other_entry": {"agent": mock_agent},
        }

        result = await async_unload_entry(mock_hass, mock_config_entry)

        assert result is True
        # Services should NOT be removed since another entry exists
        mock_remove_services.assert_not_called()


class TestServiceHandlers:
    """Test service handler functions."""

    async def test_handle_process_service(self, mock_hass, mock_agent):
        """Test process service handler."""
        entry_id = "test_entry"
        mock_hass.data[DOMAIN] = {entry_id: {"agent": mock_agent}}

        await async_setup_services(mock_hass, entry_id)

        # Get the registered handler
        process_handler = None
        for call in mock_hass.services.async_register.call_args_list:
            if call[0][1] == "process":
                process_handler = call[0][2]
                break

        assert process_handler is not None

        # Create a service call
        service_call = MagicMock(spec=ServiceCall)
        service_call.data = {
            "text": "Turn on the lights",
            "conversation_id": "conv_123",
            "user_id": "user_456",
        }

        result = await process_handler(service_call)

        assert result["response"] == "Test response"
        assert result["conversation_id"] == "conv_123"
        mock_agent.process_message.assert_called_once_with(
            text="Turn on the lights",
            conversation_id="conv_123",
            user_id="user_456",
        )

    async def test_handle_process_service_with_entry_id(self, mock_hass, mock_agent):
        """Test process service with specific entry_id."""
        entry_id = "test_entry"
        other_agent = MagicMock()
        other_agent.process_message = AsyncMock(return_value="Other response")

        mock_hass.data[DOMAIN] = {
            entry_id: {"agent": mock_agent},
            "other_entry": {"agent": other_agent},
        }

        await async_setup_services(mock_hass, entry_id)

        process_handler = None
        for call in mock_hass.services.async_register.call_args_list:
            if call[0][1] == "process":
                process_handler = call[0][2]
                break

        service_call = MagicMock(spec=ServiceCall)
        service_call.data = {
            "text": "Test",
            "entry_id": "other_entry",
        }

        result = await process_handler(service_call)

        assert result["response"] == "Other response"
        other_agent.process_message.assert_called_once()
        mock_agent.process_message.assert_not_called()

    async def test_handle_process_service_error(self, mock_hass, mock_agent):
        """Test process service handles errors."""
        entry_id = "test_entry"
        mock_agent.process_message.side_effect = Exception("Processing error")
        mock_hass.data[DOMAIN] = {entry_id: {"agent": mock_agent}}

        await async_setup_services(mock_hass, entry_id)

        process_handler = None
        for call in mock_hass.services.async_register.call_args_list:
            if call[0][1] == "process":
                process_handler = call[0][2]
                break

        service_call = MagicMock(spec=ServiceCall)
        service_call.data = {"text": "Test"}

        with pytest.raises(Exception, match="Processing error"):
            await process_handler(service_call)

    async def test_handle_clear_history_service(self, mock_hass, mock_agent):
        """Test clear_history service handler."""
        entry_id = "test_entry"
        mock_hass.data[DOMAIN] = {entry_id: {"agent": mock_agent}}

        await async_setup_services(mock_hass, entry_id)

        clear_handler = None
        for call in mock_hass.services.async_register.call_args_list:
            if call[0][1] == "clear_history":
                clear_handler = call[0][2]
                break

        assert clear_handler is not None

        service_call = MagicMock(spec=ServiceCall)
        service_call.data = {"conversation_id": "conv_123"}

        await clear_handler(service_call)

        mock_agent.clear_history.assert_called_once_with("conv_123")

    async def test_handle_clear_history_all_conversations(self, mock_hass, mock_agent):
        """Test clearing all conversation history."""
        entry_id = "test_entry"
        mock_hass.data[DOMAIN] = {entry_id: {"agent": mock_agent}}

        await async_setup_services(mock_hass, entry_id)

        clear_handler = None
        for call in mock_hass.services.async_register.call_args_list:
            if call[0][1] == "clear_history":
                clear_handler = call[0][2]
                break

        service_call = MagicMock(spec=ServiceCall)
        service_call.data = {}

        await clear_handler(service_call)

        mock_agent.clear_history.assert_called_once_with(None)

    async def test_handle_reload_context_service(self, mock_hass, mock_agent):
        """Test reload_context service handler."""
        entry_id = "test_entry"
        mock_hass.data[DOMAIN] = {entry_id: {"agent": mock_agent}}

        await async_setup_services(mock_hass, entry_id)

        reload_handler = None
        for call in mock_hass.services.async_register.call_args_list:
            if call[0][1] == "reload_context":
                reload_handler = call[0][2]
                break

        assert reload_handler is not None

        service_call = MagicMock(spec=ServiceCall)
        service_call.data = {}

        await reload_handler(service_call)

        mock_agent.reload_context.assert_called_once()

    async def test_handle_execute_tool_service(self, mock_hass, mock_agent):
        """Test execute_tool service handler."""
        entry_id = "test_entry"
        mock_hass.data[DOMAIN] = {entry_id: {"agent": mock_agent}}

        await async_setup_services(mock_hass, entry_id)

        execute_handler = None
        for call in mock_hass.services.async_register.call_args_list:
            if call[0][1] == "execute_tool":
                execute_handler = call[0][2]
                break

        assert execute_handler is not None

        service_call = MagicMock(spec=ServiceCall)
        service_call.data = {
            "tool_name": "ha_control",
            "parameters": {"entity_id": "light.living_room", "action": "turn_on"},
        }

        result = await execute_handler(service_call)

        assert result["tool_name"] == "ha_control"
        assert result["result"] == {"result": "success"}
        mock_agent.execute_tool_debug.assert_called_once()

    async def test_handle_reindex_entities_service(self, mock_hass, mock_vector_manager):
        """Test reindex_entities service handler."""
        entry_id = "test_entry"
        mock_hass.data[DOMAIN] = {entry_id: {"vector_manager": mock_vector_manager}}

        await async_setup_services(mock_hass, entry_id)

        reindex_handler = None
        for call in mock_hass.services.async_register.call_args_list:
            if call[0][1] == "reindex_entities":
                reindex_handler = call[0][2]
                break

        assert reindex_handler is not None

        service_call = MagicMock(spec=ServiceCall)
        service_call.data = {}

        result = await reindex_handler(service_call)

        assert result["indexed"] == 10
        mock_vector_manager.async_reindex_all_entities.assert_called_once()

    async def test_handle_reindex_entities_no_manager(self, mock_hass):
        """Test reindex_entities when vector manager not enabled."""
        entry_id = "test_entry"
        mock_hass.data[DOMAIN] = {entry_id: {}}

        await async_setup_services(mock_hass, entry_id)

        reindex_handler = None
        for call in mock_hass.services.async_register.call_args_list:
            if call[0][1] == "reindex_entities":
                reindex_handler = call[0][2]
                break

        service_call = MagicMock(spec=ServiceCall)
        service_call.data = {}

        result = await reindex_handler(service_call)

        assert result["error"] == "Vector DB Manager not enabled"

    async def test_handle_index_entity_service(self, mock_hass, mock_vector_manager):
        """Test index_entity service handler."""
        entry_id = "test_entry"
        mock_hass.data[DOMAIN] = {entry_id: {"vector_manager": mock_vector_manager}}

        await async_setup_services(mock_hass, entry_id)

        index_handler = None
        for call in mock_hass.services.async_register.call_args_list:
            if call[0][1] == "index_entity":
                index_handler = call[0][2]
                break

        assert index_handler is not None

        service_call = MagicMock(spec=ServiceCall)
        service_call.data = {"entity_id": "light.living_room"}

        result = await index_handler(service_call)

        assert result["entity_id"] == "light.living_room"
        assert result["status"] == "indexed"
        mock_vector_manager.async_index_entity.assert_called_once_with("light.living_room")

    async def test_handle_index_entity_no_entity_id(self, mock_hass):
        """Test index_entity without entity_id."""
        entry_id = "test_entry"
        mock_hass.data[DOMAIN] = {entry_id: {}}

        await async_setup_services(mock_hass, entry_id)

        index_handler = None
        for call in mock_hass.services.async_register.call_args_list:
            if call[0][1] == "index_entity":
                index_handler = call[0][2]
                break

        service_call = MagicMock(spec=ServiceCall)
        service_call.data = {}

        result = await index_handler(service_call)

        assert result["error"] == "entity_id is required"

    async def test_handle_list_memories_service(self, mock_hass, mock_memory_manager):
        """Test list_memories service handler."""
        entry_id = "test_entry"
        mock_memory_manager.list_all_memories = AsyncMock(
            return_value=[
                {
                    "id": "mem1",
                    "type": "fact",
                    "content": "Test memory",
                    "importance": 0.7,
                    "extracted_at": "2024-01-01T00:00:00",
                    "last_accessed": "2024-01-02T00:00:00",
                    "source_conversation_id": "conv1",
                }
            ]
        )
        mock_hass.data[DOMAIN] = {entry_id: {"memory_manager": mock_memory_manager}}

        await async_setup_services(mock_hass, entry_id)

        list_handler = None
        for call in mock_hass.services.async_register.call_args_list:
            if call[0][1] == "list_memories":
                list_handler = call[0][2]
                break

        assert list_handler is not None

        service_call = MagicMock(spec=ServiceCall)
        service_call.data = {"limit": 10, "memory_type": "fact"}

        result = await list_handler(service_call)

        assert result["total"] == 1
        assert result["memories"][0]["id"] == "mem1"
        assert result["memories"][0]["type"] == "fact"
        mock_memory_manager.list_all_memories.assert_called_once_with(limit=10, memory_type="fact")

    async def test_handle_list_memories_no_manager(self, mock_hass):
        """Test list_memories when memory manager not enabled."""
        entry_id = "test_entry"
        mock_hass.data[DOMAIN] = {entry_id: {}}

        await async_setup_services(mock_hass, entry_id)

        list_handler = None
        for call in mock_hass.services.async_register.call_args_list:
            if call[0][1] == "list_memories":
                list_handler = call[0][2]
                break

        service_call = MagicMock(spec=ServiceCall)
        service_call.data = {}

        result = await list_handler(service_call)

        assert result["error"] == "Memory Manager not enabled"
        assert result["total"] == 0

    async def test_handle_delete_memory_service(self, mock_hass, mock_memory_manager):
        """Test delete_memory service handler."""
        entry_id = "test_entry"
        mock_hass.data[DOMAIN] = {entry_id: {"memory_manager": mock_memory_manager}}

        await async_setup_services(mock_hass, entry_id)

        delete_handler = None
        for call in mock_hass.services.async_register.call_args_list:
            if call[0][1] == "delete_memory":
                delete_handler = call[0][2]
                break

        assert delete_handler is not None

        service_call = MagicMock(spec=ServiceCall)
        service_call.data = {"memory_id": "mem_123"}

        await delete_handler(service_call)

        mock_memory_manager.delete_memory.assert_called_once_with("mem_123")

    async def test_handle_delete_memory_no_manager(self, mock_hass):
        """Test delete_memory when memory manager not enabled."""
        entry_id = "test_entry"
        mock_hass.data[DOMAIN] = {entry_id: {}}

        await async_setup_services(mock_hass, entry_id)

        delete_handler = None
        for call in mock_hass.services.async_register.call_args_list:
            if call[0][1] == "delete_memory":
                delete_handler = call[0][2]
                break

        service_call = MagicMock(spec=ServiceCall)
        service_call.data = {"memory_id": "mem_123"}

        # Should return early without error
        await delete_handler(service_call)

    async def test_handle_clear_memories_service(self, mock_hass, mock_memory_manager):
        """Test clear_memories service handler with confirmation."""
        entry_id = "test_entry"
        mock_memory_manager.clear_all_memories = AsyncMock(return_value=25)
        mock_hass.data[DOMAIN] = {entry_id: {"memory_manager": mock_memory_manager}}

        await async_setup_services(mock_hass, entry_id)

        clear_handler = None
        for call in mock_hass.services.async_register.call_args_list:
            if call[0][1] == "clear_memories":
                clear_handler = call[0][2]
                break

        assert clear_handler is not None

        service_call = MagicMock(spec=ServiceCall)
        service_call.data = {"confirm": True}

        result = await clear_handler(service_call)

        assert result["deleted_count"] == 25
        mock_memory_manager.clear_all_memories.assert_called_once()

    async def test_handle_clear_memories_without_confirmation(self, mock_hass):
        """Test clear_memories without confirmation fails."""
        entry_id = "test_entry"
        mock_memory_manager = MagicMock()
        mock_hass.data[DOMAIN] = {entry_id: {"memory_manager": mock_memory_manager}}

        await async_setup_services(mock_hass, entry_id)

        clear_handler = None
        for call in mock_hass.services.async_register.call_args_list:
            if call[0][1] == "clear_memories":
                clear_handler = call[0][2]
                break

        service_call = MagicMock(spec=ServiceCall)
        service_call.data = {"confirm": False}

        result = await clear_handler(service_call)

        assert result["error"] == "confirmation_required"
        assert result["deleted_count"] == 0

    async def test_handle_search_memories_service(self, mock_hass, mock_memory_manager):
        """Test search_memories service handler."""
        entry_id = "test_entry"
        mock_memory_manager.search_memories = AsyncMock(
            return_value=[
                {
                    "id": "mem1",
                    "type": "preference",
                    "content": "User likes warm temperature",
                    "importance": 0.8,
                    "relevance_score": 0.95,
                }
            ]
        )
        mock_hass.data[DOMAIN] = {entry_id: {"memory_manager": mock_memory_manager}}

        await async_setup_services(mock_hass, entry_id)

        search_handler = None
        for call in mock_hass.services.async_register.call_args_list:
            if call[0][1] == "search_memories":
                search_handler = call[0][2]
                break

        assert search_handler is not None

        service_call = MagicMock(spec=ServiceCall)
        service_call.data = {
            "query": "temperature preferences",
            "limit": 5,
            "min_importance": 0.5,
        }

        result = await search_handler(service_call)

        assert result["total"] == 1
        assert result["memories"][0]["relevance_score"] == 0.95
        mock_memory_manager.search_memories.assert_called_once_with(
            query="temperature preferences",
            top_k=5,
            min_importance=0.5,
        )

    async def test_handle_search_memories_defaults(self, mock_hass, mock_memory_manager):
        """Test search_memories with default parameters."""
        entry_id = "test_entry"
        mock_memory_manager.search_memories = AsyncMock(return_value=[])
        mock_hass.data[DOMAIN] = {entry_id: {"memory_manager": mock_memory_manager}}

        await async_setup_services(mock_hass, entry_id)

        search_handler = None
        for call in mock_hass.services.async_register.call_args_list:
            if call[0][1] == "search_memories":
                search_handler = call[0][2]
                break

        service_call = MagicMock(spec=ServiceCall)
        service_call.data = {"query": "test"}

        await search_handler(service_call)

        mock_memory_manager.search_memories.assert_called_once_with(
            query="test",
            top_k=10,  # default
            min_importance=0.0,  # default
        )

    async def test_handle_add_memory_service(self, mock_hass, mock_memory_manager):
        """Test add_memory service handler."""
        entry_id = "test_entry"
        mock_hass.data[DOMAIN] = {entry_id: {"memory_manager": mock_memory_manager}}

        await async_setup_services(mock_hass, entry_id)

        add_handler = None
        for call in mock_hass.services.async_register.call_args_list:
            if call[0][1] == "add_memory":
                add_handler = call[0][2]
                break

        assert add_handler is not None

        service_call = MagicMock(spec=ServiceCall)
        service_call.data = {
            "content": "User prefers lights at 50%",
            "type": "preference",
            "importance": 0.7,
        }

        result = await add_handler(service_call)

        assert result["memory_id"] == "mem_123"
        mock_memory_manager.add_memory.assert_called_once()

        # Verify the call arguments
        call_args = mock_memory_manager.add_memory.call_args
        assert call_args[1]["content"] == "User prefers lights at 50%"
        assert call_args[1]["memory_type"] == "preference"
        assert call_args[1]["importance"] == 0.7
        assert call_args[1]["conversation_id"] is None
        assert call_args[1]["metadata"]["extraction_method"] == "manual_service"

    async def test_handle_add_memory_with_defaults(self, mock_hass, mock_memory_manager):
        """Test add_memory with default values."""
        entry_id = "test_entry"
        mock_hass.data[DOMAIN] = {entry_id: {"memory_manager": mock_memory_manager}}

        await async_setup_services(mock_hass, entry_id)

        add_handler = None
        for call in mock_hass.services.async_register.call_args_list:
            if call[0][1] == "add_memory":
                add_handler = call[0][2]
                break

        service_call = MagicMock(spec=ServiceCall)
        service_call.data = {"content": "Test fact"}

        result = await add_handler(service_call)

        assert result["memory_id"] == "mem_123"

        # Verify defaults were used
        call_args = mock_memory_manager.add_memory.call_args
        assert call_args[1]["memory_type"] == "fact"  # default
        assert call_args[1]["importance"] == 0.5  # default


class TestAsyncRemoveServices:
    """Test async_remove_services function."""

    async def test_remove_all_services(self, mock_hass):
        """Test removing all services."""
        # Simulate all services being registered
        mock_hass.services.has_service.return_value = True

        await async_remove_services(mock_hass)

        # Verify all services were removed
        expected_services = [
            "process",
            "clear_history",
            "reload_context",
            "execute_tool",
            "reindex_entities",
            "index_entity",
            "list_memories",
            "delete_memory",
            "clear_memories",
            "search_memories",
            "add_memory",
            "clear_conversation",
        ]

        for service in expected_services:
            mock_hass.services.async_remove.assert_any_call(DOMAIN, service)

        assert mock_hass.services.async_remove.call_count == len(expected_services)

    async def test_remove_services_when_none_registered(self, mock_hass):
        """Test removing services when none are registered."""
        mock_hass.services.has_service.return_value = False

        await async_remove_services(mock_hass)

        # Should not try to remove anything
        mock_hass.services.async_remove.assert_not_called()

    async def test_remove_some_services(self, mock_hass):
        """Test removing only some services that are registered."""

        def has_service_side_effect(domain, service):
            return service in ["process", "clear_history"]

        mock_hass.services.has_service.side_effect = has_service_side_effect

        await async_remove_services(mock_hass)

        # Only registered services should be removed
        assert mock_hass.services.async_remove.call_count == 2
        mock_hass.services.async_remove.assert_any_call(DOMAIN, "process")
        mock_hass.services.async_remove.assert_any_call(DOMAIN, "clear_history")


class TestServiceRegistration:
    """Test service registration logic."""

    async def test_services_registered_only_once(self, mock_hass):
        """Test that services are only registered once."""
        entry_id = "test_entry"
        mock_hass.data[DOMAIN] = {entry_id: {}}

        # First setup
        await async_setup_services(mock_hass, entry_id)

        first_call_count = mock_hass.services.async_register.call_count

        # Second setup - has_service should now return True
        mock_hass.services.has_service.return_value = True
        await async_setup_services(mock_hass, "another_entry")

        # No new services should be registered
        assert mock_hass.services.async_register.call_count == first_call_count

    async def test_all_services_registered(self, mock_hass):
        """Test that all expected services are registered."""
        entry_id = "test_entry"
        mock_hass.data[DOMAIN] = {entry_id: {}}

        await async_setup_services(mock_hass, entry_id)

        expected_services = [
            "process",
            "clear_history",
            "reload_context",
            "execute_tool",
            "reindex_entities",
            "index_entity",
            "list_memories",
            "delete_memory",
            "clear_memories",
            "search_memories",
            "add_memory",
            "clear_conversation",
        ]

        registered_services = [
            call[0][1] for call in mock_hass.services.async_register.call_args_list
        ]

        for service in expected_services:
            assert service in registered_services


class TestServiceErrorHandling:
    """Test error handling in service handlers."""

    async def test_agent_not_found_error(self, mock_hass):
        """Test error when agent is not found."""
        entry_id = "test_entry"
        mock_hass.data[DOMAIN] = {entry_id: {}}  # No agent

        await async_setup_services(mock_hass, entry_id)

        process_handler = None
        for call in mock_hass.services.async_register.call_args_list:
            if call[0][1] == "process":
                process_handler = call[0][2]
                break

        service_call = MagicMock(spec=ServiceCall)
        service_call.data = {"text": "Test"}

        with pytest.raises(ValueError, match="Agent not found"):
            await process_handler(service_call)

    async def test_memory_manager_error_propagates(self, mock_hass, mock_memory_manager):
        """Test that memory manager errors are propagated."""
        entry_id = "test_entry"
        mock_memory_manager.add_memory.side_effect = Exception("Database error")
        mock_hass.data[DOMAIN] = {entry_id: {"memory_manager": mock_memory_manager}}

        await async_setup_services(mock_hass, entry_id)

        add_handler = None
        for call in mock_hass.services.async_register.call_args_list:
            if call[0][1] == "add_memory":
                add_handler = call[0][2]
                break

        service_call = MagicMock(spec=ServiceCall)
        service_call.data = {"content": "Test"}

        with pytest.raises(Exception, match="Database error"):
            await add_handler(service_call)

    async def test_vector_manager_error_propagates(self, mock_hass, mock_vector_manager):
        """Test that vector manager errors are propagated."""
        entry_id = "test_entry"
        mock_vector_manager.async_reindex_all_entities.side_effect = Exception("Index error")
        mock_hass.data[DOMAIN] = {entry_id: {"vector_manager": mock_vector_manager}}

        await async_setup_services(mock_hass, entry_id)

        reindex_handler = None
        for call in mock_hass.services.async_register.call_args_list:
            if call[0][1] == "reindex_entities":
                reindex_handler = call[0][2]
                break

        service_call = MagicMock(spec=ServiceCall)
        service_call.data = {}

        with pytest.raises(Exception, match="Index error"):
            await reindex_handler(service_call)
