# Home Agent Roadmap

**Last Updated:** 2025-12-05
**Current Version:** v0.8.3

This document tracks the roadmap for Home Agent capabilities, separate from the implementation specification in PROJECT_SPEC.md.

---

## Current Capabilities (Implemented)

### Core Agent Features
- OpenAI-compatible LLM integration (any provider)
- Home Assistant ConversationEntity implementation
- Native tool calling (ha_control, ha_query)
- Streaming response support
- Multi-turn conversation management

### Context & Memory
- Direct entity context injection
- Vector database (ChromaDB) semantic search
- Additional collections support for supplementary context
- Long-term memory system with automatic extraction
- Memory quality validation and filtering

### Tools
- `ha_control` - Device and entity control
- `ha_query` - State queries with history and wildcards
- `query_external_llm` - Delegate to external LLM
- `store_memory` / `recall_memory` - Manual memory management
- Custom tools via REST and service handlers

### Voice Assistant Integration
- Session persistence for multi-turn voice conversations
- Device/user-based session mapping
- Streaming for faster first audio response (~500ms)

---

## Roadmap

### Near-Term (Phase 6-7 Completion)

#### Reliability & Resource Management
- [ ] Conversation history auto-cleanup (7-day retention)
- [ ] Connection pooling for external services
- [ ] Graceful degradation when dependencies fail
- [ ] Simple retry logic (2 attempts, 1s delay)
- [ ] User-friendly error messages

#### HACS Submission
- [ ] Add LICENSE file (MIT)
- [ ] Update manifest.json URLs
- [ ] Create info.md for HACS UI
- [ ] Run HACS validation
- [ ] Public repository setup

---

### Medium-Term

#### MCP Server Integration (Phase 5)
- [ ] Model Context Protocol server support
- [ ] MCP handler type for custom tools
- [ ] Authentication for MCP servers
- [ ] Read-only data collection from MCP

#### Code Refactoring
- [ ] Modularize agent.py (1,826 lines → 4 focused modules)
- [ ] Extract config_flow.py schemas and validators
- [ ] Extract memory validation to dedicated MemoryValidator class

#### Enhanced Documentation
- [ ] Screenshots and visual guides
- [ ] Architecture diagrams
- [ ] Video tutorials (community contribution)

---

### Long-Term / Research

#### Multi-Agent Conversations
- Multiple specialized agents working together
- Agent handoff and coordination
- Specialized agents for specific domains (climate, security, media)

#### Proactive Automation
- Suggestions based on patterns and context
- Learning from user feedback
- Anomaly detection and alerting

#### Natural Language Automation Builder
- Create automations through conversation
- Visual builder with LLM assistance
- Template suggestions based on common patterns

#### Advanced Memory Features
- User-specific memories (per-person preferences)
- Time-aware memory recall
- Memory importance evolution based on usage

#### On-Device Model Execution
- Embedded small models for simple queries
- Reduced latency for common operations
- Privacy-preserving local processing

---

## Community Requests

*Track feature requests from the community here*

| Request | Votes | Status | Notes |
|---------|-------|--------|-------|
| - | - | - | No public release yet |

---

## Completed Milestones

### v0.8.x (Current)
- Phases 1-4 + 3.5 complete
- 400+ tests, >80% coverage
- Memory system with quality validation
- Streaming response support
- Additional ChromaDB collections

### v0.4.7-beta
- Additional collections for supplementary context
- Two-tier ranking system

### Earlier Versions
- Core agent implementation
- Tool calling framework
- Vector DB integration
- External LLM tool

---

## Design Principles

1. **Simple over Complex** - Avoid enterprise-scale patterns inappropriate for home use
2. **Privacy First** - User-controlled data retention and memory toggles
3. **Graceful Degradation** - Partial functionality better than total failure
4. **Resource Efficient** - Runs well on Raspberry Pi and NUCs
5. **Provider Agnostic** - Works with any OpenAI-compatible API

---

## Contributing

Ideas and contributions welcome! Please open an issue or discussion for:
- Feature requests
- Bug reports
- Documentation improvements
- Community integrations

---

*This roadmap is a living document and subject to change based on community feedback and development priorities.*
