"""Schema definitions for Home Agent configuration.

This module contains all voluptuous schema definitions for the configuration
flow, keeping form structure separate from business logic.
"""

from __future__ import annotations

from typing import Any

import voluptuous as vol
from homeassistant.helpers import selector

from ..const import (
    CONF_ADDITIONAL_COLLECTIONS,
    CONF_ADDITIONAL_L2_DISTANCE_THRESHOLD,
    CONF_ADDITIONAL_TOP_K,
    CONF_CONTEXT_FORMAT,
    CONF_CONTEXT_MODE,
    CONF_DEBUG_LOGGING,
    CONF_DIRECT_ENTITIES,
    CONF_EMBEDDING_KEEP_ALIVE,
    CONF_EXTERNAL_LLM_API_KEY,
    CONF_EXTERNAL_LLM_AUTO_INCLUDE_CONTEXT,
    CONF_EXTERNAL_LLM_BASE_URL,
    CONF_EXTERNAL_LLM_ENABLED,
    CONF_EXTERNAL_LLM_KEEP_ALIVE,
    CONF_EXTERNAL_LLM_MAX_TOKENS,
    CONF_EXTERNAL_LLM_MODEL,
    CONF_EXTERNAL_LLM_TEMPERATURE,
    CONF_EXTERNAL_LLM_TOOL_DESCRIPTION,
    CONF_HISTORY_ENABLED,
    CONF_HISTORY_MAX_MESSAGES,
    CONF_HISTORY_MAX_TOKENS,
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_LLM_KEEP_ALIVE,
    CONF_LLM_MAX_TOKENS,
    CONF_LLM_MODEL,
    CONF_LLM_PROXY_HEADERS,
    CONF_LLM_TEMPERATURE,
    CONF_MEMORY_COLLECTION_NAME,
    CONF_MEMORY_CONTEXT_TOP_K,
    CONF_MEMORY_ENABLED,
    CONF_MEMORY_EXTRACTION_ENABLED,
    CONF_MEMORY_EXTRACTION_LLM,
    CONF_MEMORY_MAX_MEMORIES,
    CONF_MEMORY_MIN_IMPORTANCE,
    CONF_OPENAI_API_KEY,
    CONF_PROMPT_CUSTOM_ADDITIONS,
    CONF_PROMPT_USE_DEFAULT,
    CONF_SESSION_PERSISTENCE_ENABLED,
    CONF_SESSION_TIMEOUT,
    CONF_STREAMING_ENABLED,
    CONF_THINKING_ENABLED,
    CONF_TOOLS_MAX_CALLS_PER_TURN,
    CONF_TOOLS_TIMEOUT,
    CONF_VECTOR_DB_COLLECTION,
    CONF_VECTOR_DB_EMBEDDING_BASE_URL,
    CONF_VECTOR_DB_EMBEDDING_MODEL,
    CONF_VECTOR_DB_EMBEDDING_PROVIDER,
    CONF_VECTOR_DB_HOST,
    CONF_VECTOR_DB_PORT,
    CONF_VECTOR_DB_SIMILARITY_THRESHOLD,
    CONF_VECTOR_DB_TOP_K,
    CONTEXT_FORMAT_HYBRID,
    CONTEXT_FORMAT_JSON,
    CONTEXT_FORMAT_NATURAL_LANGUAGE,
    CONTEXT_MODE_DIRECT,
    CONTEXT_MODE_VECTOR_DB,
    DEFAULT_ADDITIONAL_L2_DISTANCE_THRESHOLD,
    DEFAULT_ADDITIONAL_TOP_K,
    DEFAULT_CONTEXT_FORMAT,
    DEFAULT_CONTEXT_MODE,
    DEFAULT_DEBUG_LOGGING,
    DEFAULT_EMBEDDING_KEEP_ALIVE,
    DEFAULT_EXTERNAL_LLM_AUTO_INCLUDE_CONTEXT,
    DEFAULT_EXTERNAL_LLM_ENABLED,
    DEFAULT_EXTERNAL_LLM_KEEP_ALIVE,
    DEFAULT_EXTERNAL_LLM_MAX_TOKENS,
    DEFAULT_EXTERNAL_LLM_MODEL,
    DEFAULT_EXTERNAL_LLM_TEMPERATURE,
    DEFAULT_EXTERNAL_LLM_TOOL_DESCRIPTION,
    DEFAULT_HISTORY_ENABLED,
    DEFAULT_HISTORY_MAX_MESSAGES,
    DEFAULT_HISTORY_MAX_TOKENS,
    DEFAULT_LLM_KEEP_ALIVE,
    DEFAULT_LLM_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MEMORY_COLLECTION_NAME,
    DEFAULT_MEMORY_CONTEXT_TOP_K,
    DEFAULT_MEMORY_ENABLED,
    DEFAULT_MEMORY_EXTRACTION_ENABLED,
    DEFAULT_MEMORY_EXTRACTION_LLM,
    DEFAULT_MEMORY_MAX_MEMORIES,
    DEFAULT_MEMORY_MIN_IMPORTANCE,
    DEFAULT_NAME,
    DEFAULT_PROMPT_USE_DEFAULT,
    DEFAULT_SESSION_PERSISTENCE_ENABLED,
    DEFAULT_SESSION_TIMEOUT,
    DEFAULT_STREAMING_ENABLED,
    DEFAULT_TEMPERATURE,
    DEFAULT_THINKING_ENABLED,
    DEFAULT_TOOLS_MAX_CALLS_PER_TURN,
    DEFAULT_TOOLS_TIMEOUT,
    DEFAULT_VECTOR_DB_COLLECTION,
    DEFAULT_VECTOR_DB_EMBEDDING_BASE_URL,
    DEFAULT_VECTOR_DB_EMBEDDING_MODEL,
    DEFAULT_VECTOR_DB_EMBEDDING_PROVIDER,
    DEFAULT_VECTOR_DB_HOST,
    DEFAULT_VECTOR_DB_PORT,
    DEFAULT_VECTOR_DB_SIMILARITY_THRESHOLD,
    DEFAULT_VECTOR_DB_TOP_K,
    EMBEDDING_PROVIDER_OLLAMA,
    EMBEDDING_PROVIDER_OPENAI,
)

# OpenAI default base URL
OPENAI_BASE_URL = "https://api.openai.com/v1"


def get_user_step_schema() -> vol.Schema:
    """Get schema for initial user configuration step.

    Returns:
        Schema for basic LLM configuration
    """
    return vol.Schema(
        {
            vol.Required("name", default=DEFAULT_NAME): str,
            vol.Required(
                CONF_LLM_BASE_URL,
                default=OPENAI_BASE_URL,
            ): str,
            vol.Required(CONF_LLM_API_KEY): selector.TemplateSelector(),
            vol.Required(
                CONF_LLM_MODEL,
                default=DEFAULT_LLM_MODEL,
            ): str,
            vol.Optional(
                CONF_LLM_PROXY_HEADERS,
                description={"suggested_value": ""},
            ): str,
            vol.Optional(
                CONF_LLM_TEMPERATURE,
                default=DEFAULT_TEMPERATURE,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=2.0)),
            vol.Optional(
                CONF_LLM_MAX_TOKENS,
                default=DEFAULT_MAX_TOKENS,
            ): vol.All(vol.Coerce(int), vol.Range(min=1, max=100000)),
            vol.Optional(
                CONF_LLM_KEEP_ALIVE,
                default=DEFAULT_LLM_KEEP_ALIVE,
            ): str,
        }
    )


def get_llm_settings_schema(current_data: dict[str, Any]) -> vol.Schema:
    """Get schema for LLM settings in options flow.

    Args:
        current_data: Current configuration data

    Returns:
        Schema for LLM settings
    """
    return vol.Schema(
        {
            vol.Required(
                CONF_LLM_BASE_URL,
                default=current_data.get(CONF_LLM_BASE_URL, OPENAI_BASE_URL),
            ): str,
            vol.Required(
                CONF_LLM_API_KEY,
                default=current_data.get(CONF_LLM_API_KEY, ""),
            ): selector.TemplateSelector(),
            vol.Required(
                CONF_LLM_MODEL,
                default=current_data.get(CONF_LLM_MODEL, DEFAULT_LLM_MODEL),
            ): str,
            vol.Optional(
                CONF_LLM_PROXY_HEADERS,
                description={"suggested_value": ""},
            ): str,
            vol.Optional(
                CONF_LLM_TEMPERATURE,
                default=current_data.get(CONF_LLM_TEMPERATURE, DEFAULT_TEMPERATURE),
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=2.0)),
            vol.Optional(
                CONF_LLM_MAX_TOKENS,
                default=current_data.get(CONF_LLM_MAX_TOKENS, DEFAULT_MAX_TOKENS),
            ): vol.All(vol.Coerce(int), vol.Range(min=1, max=100000)),
            vol.Optional(
                CONF_LLM_KEEP_ALIVE,
                default=current_data.get(CONF_LLM_KEEP_ALIVE, DEFAULT_LLM_KEEP_ALIVE),
            ): str,
            vol.Optional(
                CONF_THINKING_ENABLED,
                default=current_data.get(CONF_THINKING_ENABLED, DEFAULT_THINKING_ENABLED),
            ): bool,
        }
    )


def get_context_settings_schema(
    current_options: dict[str, Any], current_data: dict[str, Any]
) -> vol.Schema:
    """Get schema for context injection settings.

    Args:
        current_options: Current option values
        current_data: Current data values

    Returns:
        Schema for context settings
    """
    return vol.Schema(
        {
            vol.Required(
                CONF_CONTEXT_MODE,
                default=current_options.get(
                    CONF_CONTEXT_MODE,
                    current_data.get(CONF_CONTEXT_MODE, DEFAULT_CONTEXT_MODE),
                ),
            ): vol.In([CONTEXT_MODE_DIRECT, CONTEXT_MODE_VECTOR_DB]),
            vol.Optional(
                CONF_CONTEXT_FORMAT,
                default=current_options.get(
                    CONF_CONTEXT_FORMAT,
                    current_data.get(CONF_CONTEXT_FORMAT, DEFAULT_CONTEXT_FORMAT),
                ),
            ): vol.In(
                [
                    CONTEXT_FORMAT_JSON,
                    CONTEXT_FORMAT_NATURAL_LANGUAGE,
                    CONTEXT_FORMAT_HYBRID,
                ]
            ),
            vol.Optional(
                CONF_DIRECT_ENTITIES,
                default=current_options.get(
                    CONF_DIRECT_ENTITIES, current_data.get(CONF_DIRECT_ENTITIES, "")
                ),
            ): str,
        }
    )


def get_vector_db_settings_schema(
    current_options: dict[str, Any],
    current_data: dict[str, Any],
    additional_collections_str: str,
) -> vol.Schema:
    """Get schema for Vector DB (ChromaDB) settings.

    Args:
        current_options: Current option values
        current_data: Current data values
        additional_collections_str: Comma-separated string of additional collections

    Returns:
        Schema for Vector DB settings
    """
    return vol.Schema(
        {
            vol.Optional(
                CONF_VECTOR_DB_HOST,
                default=current_options.get(
                    CONF_VECTOR_DB_HOST,
                    current_data.get(CONF_VECTOR_DB_HOST, DEFAULT_VECTOR_DB_HOST),
                ),
            ): str,
            vol.Optional(
                CONF_VECTOR_DB_PORT,
                default=current_options.get(
                    CONF_VECTOR_DB_PORT,
                    current_data.get(CONF_VECTOR_DB_PORT, DEFAULT_VECTOR_DB_PORT),
                ),
            ): vol.All(vol.Coerce(int), vol.Range(min=1, max=65535)),
            vol.Optional(
                CONF_VECTOR_DB_COLLECTION,
                default=current_options.get(
                    CONF_VECTOR_DB_COLLECTION,
                    current_data.get(CONF_VECTOR_DB_COLLECTION, DEFAULT_VECTOR_DB_COLLECTION),
                ),
            ): str,
            vol.Optional(
                CONF_VECTOR_DB_EMBEDDING_PROVIDER,
                default=current_options.get(
                    CONF_VECTOR_DB_EMBEDDING_PROVIDER,
                    current_data.get(
                        CONF_VECTOR_DB_EMBEDDING_PROVIDER,
                        DEFAULT_VECTOR_DB_EMBEDDING_PROVIDER,
                    ),
                ),
            ): selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=[
                        EMBEDDING_PROVIDER_OPENAI,
                        EMBEDDING_PROVIDER_OLLAMA,
                    ],
                    translation_key="embedding_provider",
                )
            ),
            vol.Optional(
                CONF_VECTOR_DB_EMBEDDING_BASE_URL,
                default=current_options.get(
                    CONF_VECTOR_DB_EMBEDDING_BASE_URL,
                    current_data.get(
                        CONF_VECTOR_DB_EMBEDDING_BASE_URL,
                        DEFAULT_VECTOR_DB_EMBEDDING_BASE_URL,
                    ),
                ),
            ): str,
            vol.Optional(
                CONF_VECTOR_DB_EMBEDDING_MODEL,
                default=current_options.get(
                    CONF_VECTOR_DB_EMBEDDING_MODEL,
                    current_data.get(
                        CONF_VECTOR_DB_EMBEDDING_MODEL,
                        DEFAULT_VECTOR_DB_EMBEDDING_MODEL,
                    ),
                ),
            ): str,
            vol.Optional(
                CONF_EMBEDDING_KEEP_ALIVE,
                default=current_options.get(
                    CONF_EMBEDDING_KEEP_ALIVE,
                    current_data.get(CONF_EMBEDDING_KEEP_ALIVE, DEFAULT_EMBEDDING_KEEP_ALIVE),
                ),
            ): str,
            vol.Optional(
                CONF_OPENAI_API_KEY,
                default=current_options.get(
                    CONF_OPENAI_API_KEY, current_data.get(CONF_OPENAI_API_KEY, "")
                ),
            ): selector.TemplateSelector(),
            vol.Optional(
                CONF_VECTOR_DB_TOP_K,
                default=current_options.get(
                    CONF_VECTOR_DB_TOP_K,
                    current_data.get(CONF_VECTOR_DB_TOP_K, DEFAULT_VECTOR_DB_TOP_K),
                ),
            ): vol.All(vol.Coerce(int), vol.Range(min=1, max=50)),
            vol.Optional(
                CONF_VECTOR_DB_SIMILARITY_THRESHOLD,
                default=current_options.get(
                    CONF_VECTOR_DB_SIMILARITY_THRESHOLD,
                    current_data.get(
                        CONF_VECTOR_DB_SIMILARITY_THRESHOLD,
                        DEFAULT_VECTOR_DB_SIMILARITY_THRESHOLD,
                    ),
                ),
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1000.0)),
            vol.Optional(
                CONF_ADDITIONAL_COLLECTIONS,
                default=additional_collections_str,
            ): str,
            vol.Optional(
                CONF_ADDITIONAL_TOP_K,
                default=current_options.get(CONF_ADDITIONAL_TOP_K, DEFAULT_ADDITIONAL_TOP_K),
            ): vol.All(vol.Coerce(int), vol.Range(min=1, max=50)),
            vol.Optional(
                CONF_ADDITIONAL_L2_DISTANCE_THRESHOLD,
                default=current_options.get(
                    CONF_ADDITIONAL_L2_DISTANCE_THRESHOLD,
                    DEFAULT_ADDITIONAL_L2_DISTANCE_THRESHOLD,
                ),
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=2000.0)),
        }
    )


def get_history_settings_schema(
    current_options: dict[str, Any], current_data: dict[str, Any]
) -> vol.Schema:
    """Get schema for conversation history settings.

    Args:
        current_options: Current option values
        current_data: Current data values

    Returns:
        Schema for history settings
    """
    return vol.Schema(
        {
            vol.Required(
                CONF_HISTORY_ENABLED,
                default=current_options.get(
                    CONF_HISTORY_ENABLED,
                    current_data.get(CONF_HISTORY_ENABLED, DEFAULT_HISTORY_ENABLED),
                ),
            ): bool,
            vol.Optional(
                CONF_HISTORY_MAX_MESSAGES,
                default=current_options.get(
                    CONF_HISTORY_MAX_MESSAGES,
                    current_data.get(CONF_HISTORY_MAX_MESSAGES, DEFAULT_HISTORY_MAX_MESSAGES),
                ),
            ): vol.All(vol.Coerce(int), vol.Range(min=1, max=100)),
            vol.Optional(
                CONF_HISTORY_MAX_TOKENS,
                default=current_options.get(
                    CONF_HISTORY_MAX_TOKENS,
                    current_data.get(CONF_HISTORY_MAX_TOKENS, DEFAULT_HISTORY_MAX_TOKENS),
                ),
            ): vol.All(vol.Coerce(int), vol.Range(min=100, max=50000)),
            vol.Required(
                CONF_SESSION_PERSISTENCE_ENABLED,
                default=current_options.get(
                    CONF_SESSION_PERSISTENCE_ENABLED,
                    current_data.get(
                        CONF_SESSION_PERSISTENCE_ENABLED,
                        DEFAULT_SESSION_PERSISTENCE_ENABLED,
                    ),
                ),
            ): bool,
            vol.Optional(
                CONF_SESSION_TIMEOUT,
                default=current_options.get(
                    CONF_SESSION_TIMEOUT,
                    current_data.get(CONF_SESSION_TIMEOUT, DEFAULT_SESSION_TIMEOUT),
                )
                // 60,  # Convert seconds to minutes for display
            ): vol.All(vol.Coerce(int), vol.Range(min=1, max=120)),
        }
    )


def get_prompt_settings_schema(
    current_options: dict[str, Any], current_data: dict[str, Any]
) -> vol.Schema:
    """Get schema for system prompt settings.

    Args:
        current_options: Current option values
        current_data: Current data values

    Returns:
        Schema for prompt settings
    """
    return vol.Schema(
        {
            vol.Required(
                CONF_PROMPT_USE_DEFAULT,
                default=current_options.get(
                    CONF_PROMPT_USE_DEFAULT,
                    current_data.get(CONF_PROMPT_USE_DEFAULT, DEFAULT_PROMPT_USE_DEFAULT),
                ),
            ): bool,
            vol.Optional(
                CONF_PROMPT_CUSTOM_ADDITIONS,
                description={
                    "suggested_value": current_options.get(
                        CONF_PROMPT_CUSTOM_ADDITIONS,
                        current_data.get(CONF_PROMPT_CUSTOM_ADDITIONS, ""),
                    )
                },
            ): selector.TemplateSelector(),
        }
    )


def get_tool_settings_schema(
    current_options: dict[str, Any], current_data: dict[str, Any]
) -> vol.Schema:
    """Get schema for tool execution settings.

    Args:
        current_options: Current option values
        current_data: Current data values

    Returns:
        Schema for tool settings
    """
    return vol.Schema(
        {
            vol.Optional(
                CONF_TOOLS_MAX_CALLS_PER_TURN,
                default=current_options.get(
                    CONF_TOOLS_MAX_CALLS_PER_TURN,
                    current_data.get(
                        CONF_TOOLS_MAX_CALLS_PER_TURN, DEFAULT_TOOLS_MAX_CALLS_PER_TURN
                    ),
                ),
            ): vol.All(vol.Coerce(int), vol.Range(min=1, max=20)),
            vol.Optional(
                CONF_TOOLS_TIMEOUT,
                default=current_options.get(
                    CONF_TOOLS_TIMEOUT,
                    current_data.get(CONF_TOOLS_TIMEOUT, DEFAULT_TOOLS_TIMEOUT),
                ),
            ): vol.All(vol.Coerce(int), vol.Range(min=5, max=300)),
        }
    )


def get_external_llm_settings_schema(
    current_options: dict[str, Any], current_data: dict[str, Any]
) -> vol.Schema:
    """Get schema for external LLM tool settings.

    Args:
        current_options: Current option values
        current_data: Current data values

    Returns:
        Schema for external LLM settings
    """
    return vol.Schema(
        {
            vol.Required(
                CONF_EXTERNAL_LLM_ENABLED,
                default=current_options.get(
                    CONF_EXTERNAL_LLM_ENABLED,
                    current_data.get(CONF_EXTERNAL_LLM_ENABLED, DEFAULT_EXTERNAL_LLM_ENABLED),
                ),
            ): bool,
            vol.Optional(
                CONF_EXTERNAL_LLM_BASE_URL,
                default=current_options.get(
                    CONF_EXTERNAL_LLM_BASE_URL,
                    current_data.get(CONF_EXTERNAL_LLM_BASE_URL, OPENAI_BASE_URL),
                ),
            ): str,
            vol.Optional(
                CONF_EXTERNAL_LLM_API_KEY,
                default=current_options.get(
                    CONF_EXTERNAL_LLM_API_KEY, current_data.get(CONF_EXTERNAL_LLM_API_KEY, "")
                ),
            ): selector.TemplateSelector(),
            vol.Optional(
                CONF_EXTERNAL_LLM_MODEL,
                default=current_options.get(
                    CONF_EXTERNAL_LLM_MODEL,
                    current_data.get(CONF_EXTERNAL_LLM_MODEL, DEFAULT_EXTERNAL_LLM_MODEL),
                ),
            ): str,
            vol.Optional(
                CONF_EXTERNAL_LLM_TEMPERATURE,
                default=current_options.get(
                    CONF_EXTERNAL_LLM_TEMPERATURE,
                    current_data.get(
                        CONF_EXTERNAL_LLM_TEMPERATURE, DEFAULT_EXTERNAL_LLM_TEMPERATURE
                    ),
                ),
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=2.0)),
            vol.Optional(
                CONF_EXTERNAL_LLM_MAX_TOKENS,
                default=current_options.get(
                    CONF_EXTERNAL_LLM_MAX_TOKENS,
                    current_data.get(CONF_EXTERNAL_LLM_MAX_TOKENS, DEFAULT_EXTERNAL_LLM_MAX_TOKENS),
                ),
            ): vol.All(vol.Coerce(int), vol.Range(min=1, max=100000)),
            vol.Optional(
                CONF_EXTERNAL_LLM_KEEP_ALIVE,
                default=current_options.get(
                    CONF_EXTERNAL_LLM_KEEP_ALIVE,
                    current_data.get(CONF_EXTERNAL_LLM_KEEP_ALIVE, DEFAULT_EXTERNAL_LLM_KEEP_ALIVE),
                ),
            ): str,
            vol.Optional(
                CONF_EXTERNAL_LLM_TOOL_DESCRIPTION,
                description={
                    "suggested_value": current_options.get(
                        CONF_EXTERNAL_LLM_TOOL_DESCRIPTION,
                        current_data.get(
                            CONF_EXTERNAL_LLM_TOOL_DESCRIPTION,
                            DEFAULT_EXTERNAL_LLM_TOOL_DESCRIPTION,
                        ),
                    )
                },
            ): selector.TemplateSelector(),
            vol.Optional(
                CONF_EXTERNAL_LLM_AUTO_INCLUDE_CONTEXT,
                default=current_options.get(
                    CONF_EXTERNAL_LLM_AUTO_INCLUDE_CONTEXT,
                    current_data.get(
                        CONF_EXTERNAL_LLM_AUTO_INCLUDE_CONTEXT,
                        DEFAULT_EXTERNAL_LLM_AUTO_INCLUDE_CONTEXT,
                    ),
                ),
            ): bool,
        }
    )


def get_memory_settings_schema(
    current_options: dict[str, Any], current_data: dict[str, Any]
) -> vol.Schema:
    """Get schema for long-term memory system settings.

    Args:
        current_options: Current option values
        current_data: Current data values

    Returns:
        Schema for memory settings
    """
    return vol.Schema(
        {
            vol.Required(
                CONF_MEMORY_ENABLED,
                default=current_options.get(
                    CONF_MEMORY_ENABLED,
                    current_data.get(CONF_MEMORY_ENABLED, DEFAULT_MEMORY_ENABLED),
                ),
            ): bool,
            vol.Required(
                CONF_MEMORY_EXTRACTION_ENABLED,
                default=current_options.get(
                    CONF_MEMORY_EXTRACTION_ENABLED,
                    current_data.get(
                        CONF_MEMORY_EXTRACTION_ENABLED,
                        DEFAULT_MEMORY_EXTRACTION_ENABLED,
                    ),
                ),
            ): bool,
            vol.Required(
                CONF_MEMORY_EXTRACTION_LLM,
                default=current_options.get(
                    CONF_MEMORY_EXTRACTION_LLM,
                    current_data.get(
                        CONF_MEMORY_EXTRACTION_LLM,
                        DEFAULT_MEMORY_EXTRACTION_LLM,
                    ),
                ),
            ): vol.In(["external", "local"]),
            vol.Optional(
                CONF_MEMORY_MAX_MEMORIES,
                default=current_options.get(
                    CONF_MEMORY_MAX_MEMORIES,
                    current_data.get(
                        CONF_MEMORY_MAX_MEMORIES,
                        DEFAULT_MEMORY_MAX_MEMORIES,
                    ),
                ),
            ): vol.All(vol.Coerce(int), vol.Range(min=10, max=1000)),
            vol.Optional(
                CONF_MEMORY_MIN_IMPORTANCE,
                default=current_options.get(
                    CONF_MEMORY_MIN_IMPORTANCE,
                    current_data.get(
                        CONF_MEMORY_MIN_IMPORTANCE,
                        DEFAULT_MEMORY_MIN_IMPORTANCE,
                    ),
                ),
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0)),
            vol.Optional(
                CONF_MEMORY_CONTEXT_TOP_K,
                default=current_options.get(
                    CONF_MEMORY_CONTEXT_TOP_K,
                    current_data.get(
                        CONF_MEMORY_CONTEXT_TOP_K,
                        DEFAULT_MEMORY_CONTEXT_TOP_K,
                    ),
                ),
            ): vol.All(vol.Coerce(int), vol.Range(min=1, max=20)),
            vol.Optional(
                CONF_MEMORY_COLLECTION_NAME,
                default=current_options.get(
                    CONF_MEMORY_COLLECTION_NAME,
                    current_data.get(
                        CONF_MEMORY_COLLECTION_NAME,
                        DEFAULT_MEMORY_COLLECTION_NAME,
                    ),
                ),
            ): str,
        }
    )


def get_debug_settings_schema(
    current_options: dict[str, Any], current_data: dict[str, Any]
) -> vol.Schema:
    """Get schema for debug and logging settings.

    Args:
        current_options: Current option values
        current_data: Current data values

    Returns:
        Schema for debug settings
    """
    return vol.Schema(
        {
            vol.Required(
                CONF_DEBUG_LOGGING,
                default=current_options.get(
                    CONF_DEBUG_LOGGING,
                    current_data.get(CONF_DEBUG_LOGGING, DEFAULT_DEBUG_LOGGING),
                ),
            ): bool,
            vol.Required(
                CONF_STREAMING_ENABLED,
                default=current_options.get(
                    CONF_STREAMING_ENABLED,
                    current_data.get(CONF_STREAMING_ENABLED, DEFAULT_STREAMING_ENABLED),
                ),
            ): bool,
        }
    )
