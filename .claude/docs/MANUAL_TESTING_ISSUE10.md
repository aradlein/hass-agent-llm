# Manual Testing Guide: Phase 4 Streaming Response Support

This guide provides step-by-step instructions for manually testing the Phase 4 streaming response support implementation.

## Prerequisites

### Required Setup:
1. **Home Assistant** (2024.1+)
2. **Wyoming Protocol TTS** integration installed and configured
   - Piper TTS recommended
   - Or any Wyoming-compatible TTS engine
3. **Voice Assistant Pipeline** created and configured
4. **Home Agent** custom component installed

### Verify Wyoming TTS Setup:

1. Navigate to Settings → Devices & Services
2. Find your Wyoming integration (e.g., "Wyoming Protocol")
3. Verify TTS engine is connected and working
4. Test TTS manually: Settings → Voice Assistants → Test

## Test 1: Enable Streaming in Home Agent

### Steps:
1. Navigate to: **Settings → Devices & Services**
2. Find **Home Agent** integration
3. Click **Configure**
4. Select **Debug Settings** from the menu
5. Toggle **"Enable Streaming Responses"** to **ON**
6. Click **Submit**

### Expected Result:
✅ Setting saved successfully
✅ Integration reloads automatically
✅ Streaming now enabled

### Verification:
- Check configuration entry options include `streaming_enabled: true`
- Check Home Agent logs for "Streaming enabled" debug message (if debug logging on)

## Test 2: Voice Assistant Pipeline Setup

### Steps:
1. Navigate to: **Settings → Voice Assistants → Assistants**
2. Create new assistant or edit existing
3. Configure pipeline:
   - **Conversation Agent**: Select **Home Agent**
   - **Text-to-Speech**: Select your Wyoming TTS engine
   - **Speech-to-Text**: Optional (any STT or none)
4. Save assistant

### Expected Result:
✅ Pipeline created with Home Agent + Wyoming TTS

## Test 3: Basic Streaming Response

### Steps:
1. Use Voice Assistant or conversation service
2. Ask a simple question: "What time is it?"
3. Observe response timing

### Expected Result:
✅ Response starts playing **within ~500ms**
✅ Audio playback begins before full text is generated
✅ No errors in logs

### Non-Streaming Comparison:
- Disable streaming in Home Agent config
- Ask same question
- Observe **5+ second delay** before audio starts
- Re-enable streaming for remaining tests

## Test 4: Streaming with Tool Calls

### Steps:
1. Ask a question requiring a tool: "Turn on the living room light"
2. Observe behavior during tool execution
3. Check events in Developer Tools → Events

### Expected Result:
✅ Response starts: "Let me..." (streaming)
✅ Tool executes (light turns on)
✅ Response continues: "I've turned on the light" (streaming resumes)
✅ Tool progress events visible:
   - `home_agent.tool.progress` (status: started)
   - `home_agent.tool.progress` (status: completed)

### Event Monitoring:
```yaml
# Developer Tools → Events → Listen to Event
Event Type: home_agent.tool.progress
```

Example event:
```json
{
  "tool_name": "ha_control",
  "tool_call_id": "call_xyz",
  "status": "started",
  "timestamp": 1234567890.123
}
```

## Test 5: Multiple Tool Calls

### Steps:
1. Ask: "Turn on the living room light and set the temperature to 72"
2. Observe response and tool execution

### Expected Result:
✅ First tool executes (light)
✅ Second tool executes (temperature)
✅ Multiple tool progress events
✅ Response continues streaming between tool calls

## Test 6: Streaming Fallback

### Steps:
1. Temporarily break streaming (e.g., stop Ollama)
2. Ask a question
3. Check events and logs

### Expected Result:
✅ `home_agent.streaming.error` event fired
✅ Automatic fallback to synchronous mode
✅ Response still generated (via fallback)
✅ Warning in logs: "Streaming failed, falling back..."

### Event Verification:
```yaml
# Developer Tools → Events → Listen to Event
Event Type: home_agent.streaming.error
```

Example event:
```json
{
  "error": "Connection refused",
  "error_type": "ConnectionError",
  "fallback": true
}
```

## Test 7: Streaming Disabled

### Steps:
1. Disable streaming in Home Agent config
2. Ask a question
3. Observe behavior

### Expected Result:
✅ Traditional synchronous response
✅ Complete response generated before TTS starts
✅ 5+ second delay to first audio
✅ No streaming events
✅ No streaming-related errors

## Test 8: Long Response Streaming

### Steps:
1. Re-enable streaming
2. Ask for a long response: "Tell me a story about a robot"
3. Observe streaming behavior

### Expected Result:
✅ Audio starts playing after ~500ms
✅ Text continues generating while audio plays
✅ Seamless transition between sentences
✅ No audio dropouts or stuttering

## Test 9: Conversation History with Streaming

### Steps:
1. Enable conversation history
2. Have a multi-turn conversation:
   - "What's the weather?" (streaming)
   - "And tomorrow?" (streaming with context)
   - "Thanks!" (streaming)

### Expected Result:
✅ All responses use streaming
✅ Context preserved across turns
✅ History includes full streamed responses

## Test 10: Performance Comparison

### Measurement:
1. **Streaming Disabled**:
   - Ask: "What can you do?"
   - Measure time from request to first audio
   - Expected: 5-10 seconds

2. **Streaming Enabled**:
   - Ask same question
   - Measure time from request to first audio
   - Expected: 0.5-1 second

### Expected Result:
✅ ~10x improvement in time-to-first-audio
✅ Streaming latency <1 second
✅ Synchronous latency >5 seconds

## Troubleshooting

### Streaming Not Working

**Symptom**: No performance improvement with streaming enabled

**Check**:
1. Verify streaming enabled in Home Agent config
2. Verify using Voice Assistant pipeline (not direct conversation.process service)
3. Verify Wyoming TTS connected
4. Check logs for streaming errors
5. Verify ChatLog has delta_listener (requires Assist Pipeline)

### Tool Calls Not Executing

**Symptom**: Tools mentioned but not executed during streaming

**Check**:
1. Verify tools are enabled in Home Agent
2. Check `home_agent.tool.progress` events
3. Check logs for tool execution errors
4. Verify entity permissions

### Audio Stuttering

**Symptom**: Audio playback is choppy during streaming

**Possible Causes**:
- TTS engine can't keep up with streaming
- Network latency to Ollama
- System resource constraints

**Solutions**:
- Use local Ollama instance
- Increase system resources
- Reduce streaming chunk size (future enhancement)

### Streaming Errors

**Symptom**: `home_agent.streaming.error` events appearing

**Check**:
1. Ollama/LLM connection stable
2. LLM supports streaming
3. Network connectivity
4. Check error details in event data

## Success Criteria

All tests should demonstrate:
- ✅ Streaming can be enabled/disabled via UI
- ✅ Streaming reduces first-audio latency by ~10x
- ✅ Tool calls work during streaming
- ✅ Tool progress events are fired
- ✅ Automatic fallback to synchronous on errors
- ✅ Conversation history preserved
- ✅ No regressions in synchronous mode

## Reporting Issues

If any test fails, please report with:
1. Test number and name
2. Steps to reproduce
3. Expected vs actual behavior
4. Relevant logs from Home Assistant
5. Event data from Developer Tools
6. Home Agent configuration (sanitized)

## Next Steps

After successful manual testing:
- Consider enabling streaming by default (optional)
- Monitor performance in production
- Collect user feedback
- Report any issues to GitHub

---

**Testing completed by**: _______________
**Date**: _______________
**HA Version**: _______________
**Home Agent Version**: _______________
**Result**: PASS / FAIL
