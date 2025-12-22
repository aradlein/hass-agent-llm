"""Unit tests for MemoryValidator class.

Tests the declarative validation logic including:
- Word count validation (minimum meaningful words)
- Low-value prefix detection
- Low-value pattern detection
- Importance threshold checking
- Transient state detection with context awareness
- Batch validation and statistics
"""

import pytest

from custom_components.home_agent.memory.validator import MemoryValidator


class TestMemoryValidatorInitialization:
    """Test MemoryValidator initialization and configuration."""

    def test_default_initialization(self):
        """Test default configuration values."""
        validator = MemoryValidator()
        assert validator.min_word_count == 10
        assert validator.min_importance == 0.4

    def test_custom_initialization(self):
        """Test custom configuration values."""
        validator = MemoryValidator(min_word_count=15, min_importance=0.6)
        assert validator.min_word_count == 15
        assert validator.min_importance == 0.6

    def test_declarative_patterns_defined(self):
        """Test that declarative pattern lists are defined."""
        assert len(MemoryValidator.LOW_VALUE_PREFIXES) > 0
        assert len(MemoryValidator.LOW_VALUE_PATTERNS) > 0
        assert len(MemoryValidator.TRANSIENT_STATE_PATTERNS) > 0
        assert len(MemoryValidator.TEMPORAL_CONTEXT_WORDS) > 0


class TestWordCountValidation:
    """Test word count validation."""

    def test_rejects_too_short_content(self):
        """Test that content with too few meaningful words is rejected."""
        validator = MemoryValidator()
        memory = {
            "content": "Light is on",  # Only 2 meaningful words (>2 chars)
            "importance": 0.8,
        }
        is_valid, reason = validator.validate(memory)
        assert not is_valid
        assert reason.startswith("too_short:")

    def test_accepts_sufficient_word_count(self):
        """Test that content with enough words is accepted."""
        validator = MemoryValidator()
        memory = {
            "content": "User prefers the bedroom temperature at exactly 68 degrees Fahrenheit for comfortable sleeping conditions",
            "importance": 0.8,
        }
        is_valid, reason = validator.validate(memory)
        assert is_valid
        assert reason == ""

    def test_counts_only_meaningful_words(self):
        """Test that only words >2 chars are counted."""
        validator = MemoryValidator(min_word_count=5)
        # "a", "is", "on", "in" are not counted (<=2 chars)
        # Only "User", "prefers", "light", "the", "kitchen" are counted (5 words)
        memory = {
            "content": "User prefers light in the kitchen",
            "importance": 0.8,
        }
        is_valid, reason = validator.validate(memory)
        assert is_valid

    def test_custom_min_word_count(self):
        """Test custom minimum word count threshold."""
        validator = MemoryValidator(min_word_count=5)
        memory = {
            "content": "User likes warm temperatures",  # 4 meaningful words
            "importance": 0.8,
        }
        is_valid, reason = validator.validate(memory)
        assert not is_valid
        assert "too_short" in reason


class TestLowValuePrefixDetection:
    """Test low-value prefix detection."""

    def test_rejects_there_is_no_prefix(self):
        """Test rejection of 'there is no' prefix."""
        validator = MemoryValidator()
        memory = {
            "content": "There is no sensor in the bedroom to monitor humidity levels throughout the night",
            "importance": 0.8,
        }
        is_valid, reason = validator.validate(memory)
        assert not is_valid
        assert "low_value_prefix:there is no" in reason

    def test_rejects_there_are_no_prefix(self):
        """Test rejection of 'there are no' prefix."""
        validator = MemoryValidator()
        memory = {
            "content": "There are no lights configured properly inside the basement area throughout the entire house",
            "importance": 0.8,
        }
        is_valid, reason = validator.validate(memory)
        assert not is_valid
        assert "low_value_prefix:there are no" in reason

    def test_rejects_temporal_at_prefix(self):
        """Test rejection of temporal 'at' prefix with timestamp."""
        validator = MemoryValidator()
        memory = {
            "content": "At 8:30 PM the user mentioned they wanted lighting options discussed further later tonight",
            "importance": 0.8,
        }
        is_valid, reason = validator.validate(memory)
        assert not is_valid
        assert "temporal_at" in reason

    def test_accepts_at_prefix_without_timestamp(self):
        """Test that 'at' prefix without colon is not rejected as temporal."""
        validator = MemoryValidator()
        memory = {
            "content": "At home the user prefers to keep all the lights dimmed during evening hours",
            "importance": 0.8,
        }
        is_valid, reason = validator.validate(memory)
        assert is_valid


class TestLowValuePatternDetection:
    """Test low-value pattern detection."""

    def test_rejects_conversation_meta_patterns(self):
        """Test rejection of conversation meta-information."""
        validator = MemoryValidator()
        patterns_to_test = [
            "The conversation occurred yesterday afternoon when everyone started discussing temperature settings together",
            "We discussed the lighting preferences throughout the living room extensively during morning hours",
            "User asked about the thermostat settings multiple times during our conversation today afternoon",
            "During the conversation they talked about various automation options including lighting controls",
        ]

        for content in patterns_to_test:
            memory = {"content": content, "importance": 0.8}
            is_valid, reason = validator.validate(memory)
            assert not is_valid, f"Should reject: {content}"
            assert "low_value_pattern" in reason

    def test_rejects_negative_existence_patterns(self):
        """Test rejection of negative existence statements."""
        validator = MemoryValidator()
        patterns_to_test = [
            "The home does not have any automated blinds installed properly inside the bedroom area currently",
            "User doesn't have any smart thermostat installed properly inside their kitchen area currently",
            "There is no specific sensor available currently for monitoring the humidity levels inside bedroom",
        ]

        for content in patterns_to_test:
            memory = {"content": content, "importance": 0.8}
            is_valid, reason = validator.validate(memory)
            assert not is_valid, f"Should reject: {content}"

    def test_accepts_content_without_low_value_patterns(self):
        """Test that content without low-value patterns is accepted."""
        validator = MemoryValidator()
        memory = {
            "content": "User prefers keeping the bedroom temperature around 68 degrees Fahrenheit overnight for sleeping",
            "importance": 0.8,
        }
        is_valid, reason = validator.validate(memory)
        assert is_valid


class TestImportanceValidation:
    """Test importance score validation."""

    def test_rejects_low_importance(self):
        """Test rejection of memories with low importance scores."""
        validator = MemoryValidator()
        memory = {
            "content": "User mentioned they might want eventually changing the light color someday later perhaps",
            "importance": 0.3,  # Below 0.4 threshold
        }
        is_valid, reason = validator.validate(memory)
        assert not is_valid
        assert "low_importance" in reason

    def test_accepts_sufficient_importance(self):
        """Test acceptance of memories with sufficient importance."""
        validator = MemoryValidator()
        memory = {
            "content": "User prefers keeping the bedroom temperature around exactly 68 degrees overnight for sleeping",
            "importance": 0.5,
        }
        is_valid, reason = validator.validate(memory)
        assert is_valid

    def test_accepts_high_importance(self):
        """Test acceptance of memories with high importance."""
        validator = MemoryValidator()
        memory = {
            "content": "User has severe allergies and needs the air purifier running continuously throughout the night",
            "importance": 0.9,
        }
        is_valid, reason = validator.validate(memory)
        assert is_valid

    def test_custom_importance_threshold(self):
        """Test custom importance threshold."""
        validator = MemoryValidator(min_importance=0.6)
        memory = {
            "content": "User prefers the living room lights being dimmed down during movie time evening hours",
            "importance": 0.5,  # Below 0.6 threshold
        }
        is_valid, reason = validator.validate(memory)
        assert not is_valid
        assert "low_importance" in reason


class TestTransientStateDetection:
    """Test transient state pattern detection."""

    def test_detects_device_state_patterns(self):
        """Test detection of device state patterns."""
        validator = MemoryValidator()
        transient_contents = [
            "The kitchen light is on right now shining brightly so the user can clearly see everything",
            "Temperature is currently showing 72 degrees inside the living room area near the window",
            "The front door is closed properly and locked securely for the entire night shift period",
            "All the basement lights are off completely because nobody seems to be there anymore",
            "The thermostat status is heating the home gradually towards the target temperature setting",
        ]

        for content in transient_contents:
            memory = {"content": content, "importance": 0.8}
            is_valid, reason = validator.validate(memory)
            assert not is_valid, f"Should detect transient state: {content}"
            assert "transient_state" in reason

    def test_context_aware_birthday_exception(self):
        """Test that 'birthday is on' is not flagged as transient."""
        validator = MemoryValidator()
        memory = {
            "content": "User's birthday is on September 28th every year and they want special lighting arrangements",
            "importance": 0.8,
        }
        is_valid, reason = validator.validate(memory)
        assert is_valid, "Should not flag 'birthday is on' as transient"

    def test_context_aware_event_exception(self):
        """Test that 'event is on' date context is not flagged."""
        validator = MemoryValidator()
        memory = {
            "content": "The anniversary event is on December 15th annually and needs special preparation arrangements",
            "importance": 0.8,
        }
        is_valid, reason = validator.validate(memory)
        assert is_valid

    def test_rejects_playing_paused_stopped(self):
        """Test detection of media state patterns."""
        validator = MemoryValidator()
        patterns = [
            "The television is playing content loudly inside the living room right at the moment",
            "Media player is paused temporarily because the phone rang earlier during the show",
            "The music is stopped completely since nobody appears to be home anymore today",
        ]

        for content in patterns:
            memory = {"content": content, "importance": 0.8}
            is_valid, reason = validator.validate(memory)
            assert not is_valid, f"Should detect: {content}"


class TestValidMemoryAcceptance:
    """Test that valid memories are properly accepted."""

    def test_accepts_user_preferences(self):
        """Test acceptance of user preference memories."""
        validator = MemoryValidator()
        memories = [
            "User prefers keeping the bedroom temperature around exactly 68 degrees Fahrenheit overnight for sleeping",
            "User likes having the kitchen lights dimmed around 50 percent brightness during daytime hours",
            "User wants the living room curtains closed automatically right around sunset time every day",
        ]

        for content in memories:
            memory = {"content": content, "importance": 0.8}
            is_valid, reason = validator.validate(memory)
            assert is_valid, f"Should accept: {content}"

    def test_accepts_permanent_facts(self):
        """Test acceptance of permanent fact memories."""
        validator = MemoryValidator()
        memories = [
            "User's birthday is on September 28th 1982 according to their personal records file",
            "The kitchen has three ceiling lights installed properly directly above the island counter",
            "User works night shifts regularly from Monday through Friday every single week consistently",
        ]

        for content in memories:
            memory = {"content": content, "importance": 0.8}
            is_valid, reason = validator.validate(memory)
            assert is_valid, f"Should accept: {content}"

    def test_accepts_device_capabilities(self):
        """Test acceptance of device capability information."""
        validator = MemoryValidator()
        memory = {
            "content": "The bedroom thermostat supports heating cooling and auto modes with programmable scheduling",
            "importance": 0.7,
        }
        is_valid, reason = validator.validate(memory)
        assert is_valid


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""

    def test_handles_missing_content(self):
        """Test handling of memory without content field."""
        validator = MemoryValidator()
        memory = {"importance": 0.8, "type": "fact"}
        is_valid, reason = validator.validate(memory)
        assert not is_valid
        assert reason == "missing_content"

    def test_handles_empty_content(self):
        """Test handling of memory with empty content."""
        validator = MemoryValidator()
        memory = {"content": "", "importance": 0.8}
        is_valid, reason = validator.validate(memory)
        assert not is_valid
        assert reason == "missing_content"

    def test_handles_non_dict_memory(self):
        """Test handling of non-dictionary memory."""
        validator = MemoryValidator()
        is_valid, reason = validator.validate("not a dict")
        assert not is_valid
        assert reason == "invalid_format"

    def test_handles_missing_importance(self):
        """Test that missing importance defaults to 0.5."""
        validator = MemoryValidator()
        memory = {
            "content": "User prefers keeping the bedroom temperature around 68 degrees overnight for sleeping comfortably"
        }
        is_valid, reason = validator.validate(memory)
        assert is_valid  # Default 0.5 is >= 0.4 threshold


class TestBatchValidation:
    """Test batch validation functionality."""

    def test_validate_batch_returns_list(self):
        """Test that validate_batch returns a list of results."""
        validator = MemoryValidator()
        memories = [
            {"content": "User prefers keeping bedroom temperature around 68 degrees overnight for sleeping comfortably", "importance": 0.8},
            {"content": "Light is on", "importance": 0.5},
        ]
        results = validator.validate_batch(memories)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_validate_batch_correct_results(self):
        """Test that batch validation returns correct results."""
        validator = MemoryValidator()
        memories = [
            {"content": "User prefers keeping bedroom temperature around exactly 68 degrees overnight for sleeping comfortably", "importance": 0.8},
            {"content": "Light is on", "importance": 0.5},
            {"content": "There is no sensor installed anywhere inside the bedroom area for monitoring the temperature accurately", "importance": 0.8},
        ]
        results = validator.validate_batch(memories)

        assert results[0][0] is True  # Valid
        assert results[1][0] is False  # Too short
        assert results[2][0] is False  # Low value prefix

    def test_validate_empty_batch(self):
        """Test validation of empty batch."""
        validator = MemoryValidator()
        results = validator.validate_batch([])
        assert results == []


class TestValidationStatistics:
    """Test validation statistics functionality."""

    def test_get_validation_stats_structure(self):
        """Test that validation stats has correct structure."""
        validator = MemoryValidator()
        memories = [
            {"content": "User prefers keeping bedroom temperature around exactly 68 degrees overnight for sleeping comfortably", "importance": 0.8},
            {"content": "Light on", "importance": 0.5},
        ]
        stats = validator.get_validation_stats(memories)

        assert "total" in stats
        assert "valid" in stats
        assert "invalid" in stats
        assert "rejection_reasons" in stats

    def test_get_validation_stats_counts(self):
        """Test that validation stats counts are correct."""
        validator = MemoryValidator()
        memories = [
            {"content": "User prefers keeping bedroom temperature around exactly 68 degrees overnight for sleeping comfortably", "importance": 0.8},
            {"content": "Light on", "importance": 0.5},  # Too short
            {"content": "There is no sensor installed anywhere inside bedroom area currently", "importance": 0.8},  # Low value
        ]
        stats = validator.get_validation_stats(memories)

        assert stats["total"] == 3
        assert stats["valid"] == 1
        assert stats["invalid"] == 2

    def test_get_validation_stats_rejection_reasons(self):
        """Test that rejection reasons are tracked."""
        validator = MemoryValidator()
        memories = [
            {"content": "Short", "importance": 0.8},  # Too short
            {"content": "Also short here", "importance": 0.8},  # Too short
            {"content": "We discussed the temperature settings multiple times during our conversation today afternoon session", "importance": 0.8},  # Pattern
        ]
        stats = validator.get_validation_stats(memories)

        assert stats["invalid"] == 3
        # Should have entries for too_short and low_value_pattern
        assert len(stats["rejection_reasons"]) >= 1

    def test_empty_stats(self):
        """Test stats for empty memory list."""
        validator = MemoryValidator()
        stats = validator.get_validation_stats([])
        assert stats["total"] == 0
        assert stats["valid"] == 0
        assert stats["invalid"] == 0
        assert stats["rejection_reasons"] == {}


class TestIsTransientStateMethod:
    """Test the is_transient_state convenience method."""

    def test_detects_transient_state(self):
        """Test detection of transient states."""
        validator = MemoryValidator()
        assert validator.is_transient_state("The light is on")
        assert validator.is_transient_state("Temperature is 72")
        assert validator.is_transient_state("We discussed this")

    def test_allows_non_transient_content(self):
        """Test that non-transient content is not flagged."""
        validator = MemoryValidator()
        assert not validator.is_transient_state("User prefers 68 degrees")
        assert not validator.is_transient_state("Birthday is on September 28")

    def test_backward_compatible_with_memory_manager(self):
        """Test that method provides similar behavior to MemoryManager._is_transient_state."""
        validator = MemoryValidator()

        # These should be detected as transient
        transient_samples = [
            "The kitchen lights are on",
            "Door is closed",
            "Thermostat status is heating",
            "We talked about lights",
            "There is no sensor",
        ]

        for sample in transient_samples:
            assert validator.is_transient_state(sample), f"Should detect: {sample}"

        # These should NOT be detected as transient
        non_transient_samples = [
            "User prefers temperature at 68",
            "Birthday is on September 28",
        ]

        for sample in non_transient_samples:
            assert not validator.is_transient_state(sample), f"Should not flag: {sample}"


class TestExpandedTransientPatterns:
    """Test the expanded transient state patterns for time, weather, date, and presence."""

    def test_detects_current_time_patterns(self):
        """Test detection of current time patterns."""
        validator = MemoryValidator()
        time_patterns = [
            "The current time is 10:30 PM and the lights need adjustment",
            "The time is 8:45 AM so we should start the morning routine",
            "It is currently 3:00 PM in the afternoon during the workday",
            "It's currently late evening according to the system clock here",
        ]

        for content in time_patterns:
            assert validator.is_transient_state(content), f"Should detect time pattern: {content}"

    def test_detects_weather_patterns(self):
        """Test detection of weather-related patterns."""
        validator = MemoryValidator()
        weather_patterns = [
            "The weather is sunny outside today in the neighborhood",
            "It's raining heavily outside so keep the windows closed",
            "It is raining and the humidity levels are rising fast",
            "It's snowing outside which means the roads are getting icy",
            "The forecast is showing rain for the next few days",
            "The forecast shows thunderstorms arriving later this evening",
            "Temperature outside is 85 degrees Fahrenheit today",
            "The outside temperature is quite cold at 32 degrees",
            "The humidity outside is very high at 90 percent currently",
            "The wind speed is 25 mph from the northwest direction",
        ]

        for content in weather_patterns:
            assert validator.is_transient_state(content), f"Should detect weather pattern: {content}"

    def test_detects_current_day_patterns(self):
        """Test detection of current date/day patterns."""
        validator = MemoryValidator()
        day_patterns = [
            "Today is Monday and the weekly cleaning schedule starts",
            "It's Tuesday which means the garbage needs collection",
            "It is Wednesday and the user works from home today",
            "This week has been particularly busy with many meetings",
            "This month the electricity bill was higher than usual",
        ]

        for content in day_patterns:
            assert validator.is_transient_state(content), f"Should detect day pattern: {content}"

    def test_detects_presence_patterns(self):
        """Test detection of presence/location patterns."""
        validator = MemoryValidator()
        presence_patterns = [
            "User is home right now working in the office area",
            "User is away and will return later this evening sometime",
            "User is not home so the thermostat should be adjusted",
            "Nobody is home currently so all lights can be turned off",
            "Someone is home based on the motion sensor detection",
            "User just arrived home from work a few minutes ago",
            "User just left the house to run some errands nearby",
        ]

        for content in presence_patterns:
            assert validator.is_transient_state(content), f"Should detect presence pattern: {content}"

    def test_allows_permanent_date_facts(self):
        """Test that permanent date facts are NOT flagged as transient."""
        validator = MemoryValidator()
        permanent_facts = [
            "User's birthday is on May 4th every single year",
            "The anniversary event is on December 25th annually",
            "User has a dentist appointment scheduled for January 15th",
            "The meeting is on the third Tuesday of every month",
            "User's vacation starts on July 1st this summer",
        ]

        for content in permanent_facts:
            assert not validator.is_transient_state(content), f"Should NOT flag permanent fact: {content}"

    def test_allows_weather_preferences(self):
        """Test that weather preferences are NOT flagged as transient."""
        validator = MemoryValidator()
        preferences = [
            "User prefers to close blinds when it gets sunny outside",
            "User wants notifications when rain is in the forecast",
            "User likes to adjust heating based on outside temperature readings",
        ]

        for content in preferences:
            assert not validator.is_transient_state(content), f"Should NOT flag preference: {content}"

    def test_detects_at_the_moment_pattern(self):
        """Test detection of 'at the moment' temporal pattern."""
        validator = MemoryValidator()
        assert validator.is_transient_state("At the moment the lights are dimmed to 50 percent")
        assert validator.is_transient_state("Right now the temperature is comfortable at 72 degrees")

    def test_full_validation_with_new_patterns(self):
        """Test full memory validation with the new transient patterns."""
        validator = MemoryValidator()

        # These should be rejected by full validation
        transient_memories = [
            {"content": "The current time is 10:29 PM according to the system clock in the living room area", "importance": 0.8},
            {"content": "It's raining outside right now so all the windows should be kept closed properly", "importance": 0.7},
            {"content": "User is currently at home working from the office area on important projects", "importance": 0.6},
            {"content": "Today is Wednesday and the weekly garbage collection schedule is starting soon", "importance": 0.5},
        ]

        for memory in transient_memories:
            is_valid, reason = validator.validate(memory)
            assert not is_valid, f"Should reject transient memory: {memory['content']}"
            assert "transient_state" in reason

        # These should be accepted
        valid_memories = [
            {"content": "User's birthday is on September 28th and they want special lighting for celebrations", "importance": 0.9},
            {"content": "User prefers the bedroom temperature at 68 degrees Fahrenheit for comfortable sleeping", "importance": 0.8},
            {"content": "User works night shifts from Monday to Friday and prefers to sleep during daytime", "importance": 0.7},
        ]

        for memory in valid_memories:
            is_valid, reason = validator.validate(memory)
            assert is_valid, f"Should accept valid memory: {memory['content']}, rejected with: {reason}"
