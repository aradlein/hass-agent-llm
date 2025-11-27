# Home Assistant Entity Services Reference (Optimized)

This document provides a comprehensive reference of all Home Assistant entity types with their **minimal required** services and attributes for LLM context. This represents the **optimized** state after removing extraneous data.

## Overview

This reference documents what SHOULD be sent to the LLM for each entity type, focusing on:
1. **Minimal useful attributes** - Only data needed for decision-making
2. **No duplicate services** - Removed homeassistant.* redundant services
3. **Domain-specific services only** - Only actionable services for each domain

## Optimization Changes

### ✅ Removed from available_services:
- `homeassistant.turn_on` (duplicate of domain service)
- `homeassistant.turn_off` (duplicate of domain service)
- `homeassistant.toggle` (duplicate of domain service)
- `homeassistant.update_entity` (not useful for user commands)
- `homeassistant.reload_config_entry` (administrative, rarely needed)

### ✅ Removed from attributes:
- **All color-related** (unless specifically needed):
  - `min_color_temp_kelvin`, `max_color_temp_kelvin`
  - `min_mireds`, `max_mireds`
  - `supported_color_modes`, `color_mode`
  - `color_temp_kelvin`, `color_temp`
  - `hs_color`, `rgb_color`, `xy_color`, `rgbw_color`, `rgbww_color`
  - `effect_list`, `effect`

- **Internal/Technical metadata**:
  - `last_changed`, `last_updated` (timestamps)
  - `device_id`, `unique_id`, `platform`, `integration`
  - `entity_id` (redundant in attributes)
  - `linkquality` (Zigbee/Z-Wave metric)
  - `update_available` (not actionable)
  - `supported_features` (bitmask)
  - `icon`, `entity_picture`, `assumed_state`, `attribution`

### Token Savings

**Estimated reduction: ~56% fewer tokens per entity (with parameter hints)**
- Before: ~115 tokens per entity (bloated with color attributes, duplicates, etc.)
- After: ~50 tokens per entity (optimized + parameter hints)
- **For 50 entities: Saves ~3,250 tokens**
- Parameter hints add ~5 tokens per entity but prevent incorrect tool calls

---

## Parameter Normalization

The `ha_control` tool automatically normalizes common parameter name mismatches to improve usability. Users and LLMs often reference **attribute names** from entity state when they should use **service parameter names**.

### Why Normalization is Needed

When users view entity state, they see attribute names (e.g., `current_position: 50` for a cover). However, Home Assistant services expect specific parameter names (e.g., `position` for `set_cover_position`). This mismatch can cause service calls to fail.

### Automatic Normalizations

The following parameter normalizations are applied automatically:

#### Cover Domain
| Attribute Name (seen in state) | Service Parameter Name | Service |
|-------------------------------|----------------------|---------|
| `current_position` | `position` | `set_cover_position` |
| `current_tilt_position` | `tilt_position` | `set_cover_tilt_position` |

**Example:**
```json
// User provides (using attribute name):
{
  "action": "set_value",
  "entity_id": "cover.kitchen_window",
  "parameters": {"current_position": 50}
}

// Automatically normalized to:
{
  "service": "cover.set_cover_position",
  "service_data": {
    "entity_id": "cover.kitchen_window",
    "position": 50
  }
}
```

#### Climate Domain
| Attribute Name | Service Parameter Name | Service | Notes |
|---------------|----------------------|---------|-------|
| `current_temperature` | `temperature` | `set_temperature` | Warning logged (user likely means target temperature) |

**Example:**
```json
// User provides (confusing read-only attribute with settable parameter):
{
  "action": "set_value",
  "entity_id": "climate.thermostat",
  "parameters": {"current_temperature": 72}
}

// Automatically normalized to (with warning):
{
  "service": "climate.set_temperature",
  "service_data": {
    "entity_id": "climate.thermostat",
    "temperature": 72
  }
}
```

### Additional Domain Handlers

The following domains now have comprehensive service mapping for `set_value` actions:

- **Media Player**: `volume_level` → `volume_set`, `source` → `select_source`
- **Humidifier**: `humidity` → `set_humidity`
- **Water Heater**: `temperature` → `set_temperature`
- **Helper Entities**: `input_text`, `input_datetime`, `number`, `select`, `text` all supported

### Usage Notes

1. Both attribute names and service parameter names are accepted
2. If the correct parameter name is already provided, normalization is skipped
3. Normalizations are logged for debugging
4. The original parameters dict is not modified (copy is created)

---

## Entity Types - Minimal Useful Data

### 1. Light

**Domain:** `light`
**Example:** `light.living_room`

**Minimal Attributes:**
```json
{
  "entity_id": "light.living_room",
  "state": "on",
  "attributes": {
    "friendly_name": "Living Room Light",
    "brightness": 180
  },
  "available_services": ["turn_on", "turn_off", "toggle"]
}
```

**Available Services:**
- `turn_on` - Turn light on (optionally with brightness)
- `turn_off` - Turn light off
- `toggle` - Toggle state

**Service Parameters:**
```json
{
  "turn_on": {
    "brightness_pct": "0-100 (preferred)",
    "brightness": "0-255 (alternative)",
    "transition": "float (seconds, optional)"
  },
  "turn_off": {
    "transition": "float (seconds, optional)"
  },
  "toggle": {}
}
```

**Note:** Color parameters removed from standard light definition. Most lights don't need color control.

---

### 2. Switch

**Domain:** `switch`
**Example:** `switch.kitchen_outlet`

**Minimal Attributes:**
```json
{
  "entity_id": "switch.kitchen_outlet",
  "state": "off",
  "attributes": {
    "friendly_name": "Kitchen Outlet"
  },
  "available_services": ["turn_on", "turn_off", "toggle"]
}
```

**Available Services:**
- `turn_on`
- `turn_off`
- `toggle`

**Service Parameters:**
```json
{
  "turn_on": {},
  "turn_off": {},
  "toggle": {}
}
```

---

### 3. Fan

**Domain:** `fan`
**Example:** `fan.bedroom_fan`

**Minimal Attributes:**
```json
{
  "entity_id": "fan.bedroom_fan",
  "state": "on",
  "attributes": {
    "friendly_name": "Bedroom Fan",
    "percentage": 66
  },
  "available_services": ["turn_on", "turn_off", "set_percentage[percentage]", "toggle", "increase_speed", "decrease_speed"]
}
```

**Available Services:**
- `turn_on`
- `turn_off`
- `set_percentage[percentage]`
- `toggle`
- `increase_speed`
- `decrease_speed`

**Service Parameters:**
```json
{
  "turn_on": {
    "percentage": "0-100 (optional)"
  },
  "set_percentage": {
    "percentage": "0-100 (required)"
  }
}
```

**Common Mappings:**
- Low: 33
- Medium: 66
- High: 100

---

### 4. Cover

**Domain:** `cover`
**Example:** `cover.living_room_blinds`

**Minimal Attributes:**
```json
{
  "entity_id": "cover.living_room_blinds",
  "state": "closed",
  "attributes": {
    "friendly_name": "Living Room Blinds",
    "current_position": 50
  },
  "available_services": ["open_cover", "close_cover", "stop_cover", "set_cover_position[position]", "toggle"]
}
```

**Available Services:**
- `open_cover`
- `close_cover`
- `stop_cover`
- `set_cover_position[position]`
- `toggle`

**Service Parameters:**
```json
{
  "open_cover": {},
  "close_cover": {},
  "stop_cover": {},
  "set_cover_position": {
    "position": "0-100 (required, 0=closed, 100=open)"
  },
  "toggle": {}
}
```

---

### 5. Lock

**Domain:** `lock`
**Example:** `lock.front_door`

**Minimal Attributes:**
```json
{
  "entity_id": "lock.front_door",
  "state": "locked",
  "attributes": {
    "friendly_name": "Front Door Lock"
  },
  "available_services": ["lock", "unlock"]
}
```

**Available Services:**
- `lock`
- `unlock`

**Service Parameters:**
```json
{
  "lock": {
    "code": "string (optional)"
  },
  "unlock": {
    "code": "string (optional)"
  }
}
```

---

### 6. Climate

**Domain:** `climate`
**Example:** `climate.thermostat`

**Minimal Attributes:**
```json
{
  "entity_id": "climate.thermostat",
  "state": "heat",
  "attributes": {
    "friendly_name": "Thermostat",
    "current_temperature": 68.0,
    "temperature": 72.0,
    "hvac_mode": "heat"
  },
  "available_services": ["set_temperature[temperature]", "set_hvac_mode[hvac_mode]", "turn_on", "turn_off"]
}
```

**Available Services:**
- `set_temperature[temperature]`
- `set_hvac_mode[hvac_mode]`
- `turn_on`
- `turn_off`

**Service Parameters:**
```json
{
  "set_temperature": {
    "temperature": "float (required)"
  },
  "set_hvac_mode": {
    "hvac_mode": "string (required: 'off', 'heat', 'cool', 'heat_cool', 'auto')"
  }
}
```

**Note:** Removed `fan_modes`, `swing_modes`, `preset_modes`, `min_temp`, `max_temp` from standard definition.

---

### 7. Media Player

**Domain:** `media_player`
**Example:** `media_player.living_room_tv`

**Minimal Attributes:**
```json
{
  "entity_id": "media_player.living_room_tv",
  "state": "playing",
  "attributes": {
    "friendly_name": "Living Room TV",
    "volume_level": 0.5
  },
  "available_services": ["turn_on", "turn_off", "media_play", "media_pause", "media_stop", "volume_set[volume_level]", "volume_up", "volume_down", "play_media[media_content_id,media_content_type]"]
}
```

**Available Services:**
- `turn_on`, `turn_off`
- `media_play`, `media_pause`, `media_stop`
- `volume_set[volume_level]`, `volume_up`, `volume_down`
- `play_media[media_content_id,media_content_type]`

**Service Parameters:**
```json
{
  "volume_set": {
    "volume_level": "0.0-1.0 (required)"
  },
  "play_media": {
    "media_content_id": "string (required)",
    "media_content_type": "string (required)"
  }
}
```

---

### 8. Sensor

**Domain:** `sensor`
**Example:** `sensor.living_room_temperature`

**Minimal Attributes:**
```json
{
  "entity_id": "sensor.living_room_temperature",
  "state": "72.5",
  "attributes": {
    "friendly_name": "Living Room Temperature",
    "unit_of_measurement": "°F"
  },
  "available_services": []
}
```

**Available Services:** None (read-only)

**Note:** Sensors are primarily for reading data.

---

### 9. Binary Sensor

**Domain:** `binary_sensor`
**Example:** `binary_sensor.front_door`

**Minimal Attributes:**
```json
{
  "entity_id": "binary_sensor.front_door",
  "state": "off",
  "attributes": {
    "friendly_name": "Front Door",
    "device_class": "door"
  },
  "available_services": []
}
```

**Available Services:** None (read-only)

**Note:** `device_class` is useful for understanding sensor type (motion, door, window, etc.)

---

### 10. Vacuum

**Domain:** `vacuum`
**Example:** `vacuum.robot_vacuum`

**Minimal Attributes:**
```json
{
  "entity_id": "vacuum.robot_vacuum",
  "state": "docked",
  "attributes": {
    "friendly_name": "Robot Vacuum",
    "battery_level": 95
  },
  "available_services": ["start", "pause", "stop", "return_to_base", "locate"]
}
```

**Available Services:**
- `start`
- `pause`
- `stop`
- `return_to_base`
- `locate`

---

### 11. Alarm Control Panel

**Domain:** `alarm_control_panel`
**Example:** `alarm_control_panel.home_alarm`

**Minimal Attributes:**
```json
{
  "entity_id": "alarm_control_panel.home_alarm",
  "state": "disarmed",
  "attributes": {
    "friendly_name": "Home Alarm"
  },
  "available_services": ["alarm_arm_home", "alarm_arm_away", "alarm_arm_night", "alarm_disarm"]
}
```

**Available Services:**
- `alarm_arm_home`
- `alarm_arm_away`
- `alarm_arm_night`
- `alarm_disarm`

**Service Parameters:**
```json
{
  "alarm_arm_home": {
    "code": "string (optional)"
  },
  "alarm_disarm": {
    "code": "string (required)"
  }
}
```

---

### 12-30. Additional Domains (Abbreviated)

| Domain | Example Entity | Key Attributes | Services |
|--------|---------------|----------------|----------|
| `water_heater` | `water_heater.tank` | friendly_name, state | turn_on, turn_off, set_temperature |
| `humidifier` | `humidifier.bedroom` | friendly_name, state | turn_on, turn_off, set_humidity |
| `valve` | `valve.water_main` | friendly_name, state | open_valve, close_valve, set_valve_position |
| `lawn_mower` | `lawn_mower.robot` | friendly_name, state | start_mowing, pause, dock |
| `button` | `button.doorbell` | friendly_name, state | press |
| `scene` | `scene.movie_time` | friendly_name, state | turn_on |
| `script` | `script.morning` | friendly_name, state | turn_on, turn_off, toggle |
| `automation` | `automation.morning` | friendly_name, state | turn_on, turn_off, toggle, trigger |
| `input_boolean` | `input_boolean.guest` | friendly_name, state | turn_on, turn_off, toggle |
| `input_number` | `input_number.offset` | friendly_name, state (value) | set_value, increment, decrement |
| `input_select` | `input_select.mode` | friendly_name, state (option) | select_option |
| `input_text` | `input_text.note` | friendly_name, state (text) | set_value |
| `input_datetime` | `input_datetime.alarm` | friendly_name, state (datetime) | set_datetime |
| `number` | `number.brightness` | friendly_name, state (value) | set_value |
| `select` | `select.theme` | friendly_name, state (option) | select_option |
| `siren` | `siren.alarm` | friendly_name, state | turn_on, turn_off |
| `camera` | `camera.front_door` | friendly_name, state | turn_on, turn_off |
| `group` | `group.all_lights` | friendly_name, state | turn_on, turn_off, toggle |

---

## Complete JSON Reference for Benchmarking

```json
{
  "entity_types": [
    {
      "domain": "light",
      "example_entity_id": "light.living_room",
      "minimal_attributes": {
        "friendly_name": "Living Room Light",
        "state": "on",
        "brightness": 180
      },
      "available_services": ["turn_on", "turn_off", "toggle"],
      "service_parameters": {
        "turn_on": {
          "brightness_pct": "0-100",
          "brightness": "0-255",
          "transition": "float"
        },
        "turn_off": {},
        "toggle": {}
      }
    },
    {
      "domain": "switch",
      "example_entity_id": "switch.kitchen_outlet",
      "minimal_attributes": {
        "friendly_name": "Kitchen Outlet",
        "state": "off"
      },
      "available_services": ["turn_on", "turn_off", "toggle"],
      "service_parameters": {
        "turn_on": {},
        "turn_off": {},
        "toggle": {}
      }
    },
    {
      "domain": "fan",
      "example_entity_id": "fan.bedroom_fan",
      "minimal_attributes": {
        "friendly_name": "Bedroom Fan",
        "state": "on",
        "percentage": 66
      },
      "available_services": ["turn_on", "turn_off", "set_percentage[percentage]", "toggle", "increase_speed", "decrease_speed"],
      "service_parameters": {
        "turn_on": {
          "percentage": "0-100"
        },
        "set_percentage": {
          "percentage": "0-100 (required)"
        }
      }
    },
    {
      "domain": "cover",
      "example_entity_id": "cover.living_room_blinds",
      "minimal_attributes": {
        "friendly_name": "Living Room Blinds",
        "state": "closed",
        "current_position": 50
      },
      "available_services": ["open_cover", "close_cover", "stop_cover", "set_cover_position[position]", "toggle"],
      "service_parameters": {
        "open_cover": {},
        "close_cover": {},
        "set_cover_position": {
          "position": "0-100 (required)"
        }
      }
    },
    {
      "domain": "lock",
      "example_entity_id": "lock.front_door",
      "minimal_attributes": {
        "friendly_name": "Front Door Lock",
        "state": "locked"
      },
      "available_services": ["lock", "unlock"],
      "service_parameters": {
        "lock": {
          "code": "string (optional)"
        },
        "unlock": {
          "code": "string (optional)"
        }
      }
    },
    {
      "domain": "climate",
      "example_entity_id": "climate.thermostat",
      "minimal_attributes": {
        "friendly_name": "Thermostat",
        "state": "heat",
        "current_temperature": 68.0,
        "temperature": 72.0,
        "hvac_mode": "heat"
      },
      "available_services": ["set_temperature", "set_hvac_mode", "turn_on", "turn_off"],
      "service_parameters": {
        "set_temperature": {
          "temperature": "float (required)"
        },
        "set_hvac_mode": {
          "hvac_mode": "string (required)"
        }
      }
    },
    {
      "domain": "media_player",
      "example_entity_id": "media_player.living_room_tv",
      "minimal_attributes": {
        "friendly_name": "Living Room TV",
        "state": "playing",
        "volume_level": 0.5
      },
      "available_services": ["turn_on", "turn_off", "media_play", "media_pause", "media_stop", "volume_set[volume_level]", "volume_up", "volume_down", "play_media[media_content_id,media_content_type]"],
      "service_parameters": {
        "volume_set": {
          "volume_level": "0.0-1.0 (required)"
        },
        "play_media": {
          "media_content_id": "string (required)",
          "media_content_type": "string (required)"
        }
      }
    },
    {
      "domain": "sensor",
      "example_entity_id": "sensor.living_room_temperature",
      "minimal_attributes": {
        "friendly_name": "Living Room Temperature",
        "state": "72.5",
        "unit_of_measurement": "°F"
      },
      "available_services": [],
      "note": "Read-only"
    },
    {
      "domain": "binary_sensor",
      "example_entity_id": "binary_sensor.front_door",
      "minimal_attributes": {
        "friendly_name": "Front Door",
        "state": "off",
        "device_class": "door"
      },
      "available_services": [],
      "note": "Read-only"
    },
    {
      "domain": "vacuum",
      "example_entity_id": "vacuum.robot_vacuum",
      "minimal_attributes": {
        "friendly_name": "Robot Vacuum",
        "state": "docked",
        "battery_level": 95
      },
      "available_services": ["start", "pause", "stop", "return_to_base", "locate"],
      "service_parameters": {}
    },
    {
      "domain": "alarm_control_panel",
      "example_entity_id": "alarm_control_panel.home",
      "minimal_attributes": {
        "friendly_name": "Home Alarm",
        "state": "disarmed"
      },
      "available_services": ["alarm_arm_home", "alarm_arm_away", "alarm_arm_night", "alarm_disarm"],
      "service_parameters": {
        "alarm_disarm": {
          "code": "string (required)"
        }
      }
    }
  ]
}
```

---

## Before vs After Comparison

### Light Entity Example

**BEFORE (Bloated - Current State):**
```json
{
  "entity_id": "light.kitchen_lights",
  "state": "on",
  "attributes": {
    "min_color_temp_kelvin": 2000,
    "max_color_temp_kelvin": 6535,
    "min_mireds": 153,
    "max_mireds": 500,
    "supported_color_modes": ["color_temp", "hs"],
    "color_mode": "hs",
    "brightness": 180,
    "color_temp_kelvin": null,
    "color_temp": null,
    "hs_color": [345, 75],
    "rgb_color": [255, 64, 112],
    "xy_color": [0.588, 0.274],
    "friendly_name": "Kitchen Lights"
  },
  "available_services": [
    "turn_on",
    "turn_off",
    "toggle",
    "homeassistant.turn_on",
    "homeassistant.turn_off",
    "homeassistant.toggle",
    "homeassistant.update_entity",
    "homeassistant.reload_config_entry"
  ]
}
```
**Token cost: ~115 tokens**

**AFTER (Optimized with Parameter Hints):**
```json
{
  "entity_id": "light.kitchen_lights",
  "state": "on",
  "attributes": {
    "friendly_name": "Kitchen Lights",
    "brightness": 180
  },
  "available_services": ["turn_on", "turn_off", "toggle"]
}
```
**Token cost: ~50 tokens (56% reduction)**

**Note:** Most light services don't have required parameters, so no hints are shown. For entities with required parameters (like media players), you'll see hints like `play_media[media_content_id,media_content_type]`.

---

## Usage in Testing and Benchmarking

### Creating Test Cases

```python
# Test that LLM uses correct services for each domain
test_cases = [
    {
        "entity_id": "cover.dining_room_blind",
        "user_input": "open the dining room blind",
        "expected_service": "open_cover",
        "expected_action": "turn_on",  # ha_control action
        "note": "LLM should use turn_on action, which maps to open_cover service"
    },
    {
        "entity_id": "lock.front_door",
        "user_input": "lock the front door",
        "expected_service": "lock",
        "expected_action": "turn_on",
        "note": "LLM should use turn_on action, which maps to lock service"
    },
    {
        "entity_id": "fan.bedroom_fan",
        "user_input": "set bedroom fan to 50%",
        "expected_service": "set_percentage",
        "expected_action": "set_value",
        "expected_params": {"percentage": 50}
    }
]
```

### Benchmarking Visibility

Test that the LLM:
1. Can see available_services for each entity
2. Understands domain-specific services (open_cover vs turn_on)
3. Correctly maps user intent to available services
4. Doesn't attempt services not in available_services list

---

## Implementation Status

### ✅ Completed Optimizations

1. **Bloat Attributes Filtering** - IMPLEMENTED
   - Expanded `BLOAT_ATTRIBUTES` from 9 to 32 attributes
   - Filters out color attributes, technical metadata, timestamps, and internal IDs
   - Applied in both `context_providers/base.py` and `context_providers/context_optimizer.py`
   - Reduces token usage by ~60% per entity

2. **Duplicate Services Removal** - VERIFIED
   - Confirmed no `homeassistant.*` duplicate services in available_services
   - Only domain-specific services are included (e.g., `light.turn_on` not `homeassistant.turn_on`)
   - Cleaner, more focused service lists for LLM context

3. **available_services Field** - IMPLEMENTED
   - Added to all code paths that return entity data
   - `context_providers/direct.py`: Added to configured entities path (line 129)
   - `tools/ha_query.py`: Added `_get_entity_services()` method and included in query results
   - LLM now sees which services are available for each entity

4. **Service Parameter Hints** - IMPLEMENTED
   - The `available_services` field now includes inline parameter hints for services with required parameters
   - Format: `service_name[required_param1,required_param2]` for services with required params
   - Example: `["turn_on", "turn_off", "media_play", "play_media[media_content_id,media_content_type]"]`
   - Queries Home Assistant service schemas to extract required parameter names
   - Limited to first 3 required parameters to minimize token usage
   - Helps LLM understand what parameters must be provided for each service

### Files Modified

1. **`context_providers/base.py`** (lines 12-57, 60-190)
   - Expanded `BLOAT_ATTRIBUTES` from 9 to 32 attributes
   - Includes color attributes, timestamps, technical metadata, internal IDs
   - Added `_add_parameter_hints_to_services()` function to extract required parameters from HA service schemas
   - Modified `get_entity_available_services()` to include parameter hints by default

2. **`context_providers/context_optimizer.py`** (2 locations)
   - Updated bloat attribute filtering to use expanded BLOAT_ATTRIBUTES
   - Applied in entity optimization logic

3. **`context_providers/direct.py`** (line 129)
   - Added `available_services` to configured entities code path
   - Ensures entities in configuration files include service information

4. **`tools/ha_query.py`** (new method: `_get_entity_services()`)
   - Added method to extract available services from Home Assistant API
   - Integrated into query results so LLM sees services when querying entities

### Implementation Notes

**Parameter Hints in Available Services:**
- The `available_services` field now includes inline hints for services with critical parameters
- Format: `service_name[param1,param2,param3]` for services with required params
- Services without required parameters appear as simple names: `turn_on`, `media_play`
- Services with required parameters show hints: `play_media[media_content_id,media_content_type]`
- Uses two sources:
  1. Hardcoded hints for critical services (play_media, set_cover_position, set_temperature, etc.)
  2. Extracted from Home Assistant's service schemas for parameters marked as required=True
- Limited to first 3 parameters to prevent token bloat
- Token cost: Approximately +300 tokens for 50 entities (8.6% of optimization savings)

**Critical Services with Hardcoded Hints:**
- Media Player: `play_media[media_content_id,media_content_type]`, `volume_set[volume_level]`
- Cover: `set_cover_position[position]`, `set_cover_tilt_position[tilt_position]`
- Climate: `set_temperature[temperature]`, `set_hvac_mode[hvac_mode]`
- Fan: `set_percentage[percentage]`
- And more for input helpers and other domains

---

**Last Updated:** 2025-11-27
**Version:** 2.0.0 (Optimized)
