# Home Agent Tool Mappings Reference

This document provides a complete reference for the domain service mappings and parameter handling used by the Home Agent LLM integration. These mappings define how generic actions (`turn_on`, `turn_off`, `toggle`, `set_value`) are translated to Home Assistant service calls.

## Table of Contents

- [Overview](#overview)
- [Domain Mappings](#domain-mappings)
  - [cover](#cover)
  - [fan](#fan)
  - [light](#light)
  - [climate](#climate)
  - [media_player](#media_player)
  - [lock](#lock)
  - [switch](#switch)
  - [humidifier](#humidifier)
  - [water_heater](#water_heater)
  - [vacuum](#vacuum)
  - [valve](#valve)
  - [lawn_mower](#lawn_mower)
  - [alarm_control_panel](#alarm_control_panel)
  - [scene](#scene)
  - [script](#script)
  - [automation](#automation)
  - [button](#button)
  - [siren](#siren)
  - [camera](#camera)
  - [group](#group)
  - [Input Helpers](#input-helpers)
- [Parameter Normalization](#parameter-normalization)
- [Auto-Conversions](#auto-conversions)
- [Special Auto-Injection Behavior](#special-auto-injection-behavior)
- [Feature Validation](#feature-validation)

---

## Overview

The Home Agent uses a unified action interface for LLM tool calls:

| Action | Description |
|--------|-------------|
| `turn_on` | Activate/enable the entity |
| `turn_off` | Deactivate/disable the entity |
| `toggle` | Switch between on/off states |
| `set_value` | Set specific attributes (brightness, temperature, position, etc.) |

These generic actions are mapped to domain-specific Home Assistant services based on the `DOMAIN_SERVICE_MAPPINGS` configuration.

---

## Domain Mappings

### cover

**Description**: Blinds, shades, garage doors, and other cover entities

#### Action Mappings

| Action | Service Called |
|--------|----------------|
| `turn_on` | `open_cover` |
| `turn_off` | `close_cover` |
| `toggle` | `toggle` |

#### Set Value Parameters

| Parameter | Service Called | Feature Required |
|-----------|----------------|------------------|
| `position` | `set_cover_position` | SET_POSITION (4) |
| `tilt_position` | `set_cover_tilt_position` | SET_TILT_POSITION (128) |

#### Parameter Normalization

- `current_position` → `position`
- `current_tilt_position` → `tilt_position`

---

### fan

**Description**: Fan entities with speed, oscillation, and direction control

#### Action Mappings

| Action | Service Called |
|--------|----------------|
| `turn_on` | `turn_on` |
| `turn_off` | `turn_off` |
| `toggle` | `toggle` |

#### Set Value Parameters

| Parameter | Service Called |
|-----------|----------------|
| `percentage` | `set_percentage` |
| `preset_mode` | `set_preset_mode` |
| `oscillating` | `oscillate` |
| `direction` | `set_direction` |

---

### light

**Description**: Light entities with brightness and color control

#### Action Mappings

| Action | Service Called |
|--------|----------------|
| `turn_on` | `turn_on` |
| `turn_off` | `turn_off` |
| `toggle` | `toggle` |
| `set_value` | `turn_on` (passes brightness/color params) |

#### Auto-Conversion

- `brightness_pct` (0-100) → `brightness` (0-255)
- Formula: `brightness = int(brightness_pct * 255 / 100)`

---

### climate

**Description**: Thermostats, air conditioners, and heating systems

#### Action Mappings

| Action | Service Called | Notes |
|--------|----------------|-------|
| `turn_on` | `set_hvac_mode` | **Auto-injects** `hvac_mode` parameter |
| `turn_off` | `set_hvac_mode` | **Auto-injects** `hvac_mode: "off"` |
| `toggle` | `toggle` | |

#### Set Value Parameters

| Parameter | Service Called |
|-----------|----------------|
| `temperature` | `set_temperature` |
| `target_temp_high` | `set_temperature` |
| `target_temp_low` | `set_temperature` |
| `hvac_mode` | `set_hvac_mode` |
| `fan_mode` | `set_fan_mode` |
| `preset_mode` | `set_preset_mode` |
| `swing_mode` | `set_swing_mode` |
| `humidity` | `set_humidity` |

#### Parameter Normalization

- `current_temperature` → `temperature` (with warning log)

#### Special Behavior

For `turn_on` without explicit `hvac_mode`, the system auto-selects the best mode from the entity's available modes in this priority order:
1. `heat_cool`
2. `auto`
3. `heat`
4. `cool`
5. First non-"off" mode available

---

### media_player

**Description**: Media playback devices and speakers

#### Action Mappings

| Action | Service Called |
|--------|----------------|
| `turn_on` | `turn_on` |
| `turn_off` | `turn_off` |
| `toggle` | `toggle` |
| `media_pause` | `media_pause` |
| `media_play` | `media_play` |
| `media_stop` | `media_stop` |
| `media_next_track` | `media_next_track` |
| `media_previous_track` | `media_previous_track` |
| `play_media` | `play_media` |

#### Set Value Parameters

| Parameter | Service Called | Notes |
|-----------|----------------|-------|
| `volume_level` | `volume_set` | |
| `percentage` | `volume_set` | Alias for volume control |
| `is_volume_muted` | `volume_mute` | |
| `source` | `select_source` | |
| `sound_mode` | `select_sound_mode` | |
| `media_content_id` | `play_media` | |
| `media_content_type` | `play_media` | For playlist/content type |
| `shuffle` | `shuffle_set` | |
| `repeat` | `repeat_set` | |

---

### lock

**Description**: Door locks and similar security devices

#### Action Mappings

| Action | Service Called |
|--------|----------------|
| `turn_on` | `lock` |
| `turn_off` | `unlock` |
| `toggle` | `toggle` |

---

### switch

**Description**: Simple on/off switches

#### Action Mappings

| Action | Service Called |
|--------|----------------|
| `turn_on` | `turn_on` |
| `turn_off` | `turn_off` |
| `toggle` | `toggle` |

---

### humidifier

**Description**: Humidifier and dehumidifier devices

#### Action Mappings

| Action | Service Called |
|--------|----------------|
| `turn_on` | `turn_on` |
| `turn_off` | `turn_off` |
| `toggle` | `toggle` |

#### Set Value Parameters

| Parameter | Service Called |
|-----------|----------------|
| `humidity` | `set_humidity` |

---

### water_heater

**Description**: Water heating systems, ovens, fridges, freezers

#### Action Mappings

| Action | Service Called |
|--------|----------------|
| `turn_on` | `turn_on` |
| `turn_off` | `turn_off` |

#### Set Value Parameters

| Parameter | Service Called |
|-----------|----------------|
| `temperature` | `set_temperature` |
| `operation_mode` | `set_operation_mode` |

---

### vacuum

**Description**: Robotic vacuum cleaners

#### Action Mappings

| Action | Service Called | Notes |
|--------|----------------|-------|
| `turn_on` | `start` | Starts cleaning |
| `turn_off` | `return_to_base` | Returns to dock |
| `toggle` | `toggle` | |

---

### valve

**Description**: Water valves and similar flow control devices

#### Action Mappings

| Action | Service Called |
|--------|----------------|
| `turn_on` | `open_valve` |
| `turn_off` | `close_valve` |
| `toggle` | `toggle` |

#### Set Value Parameters

| Parameter | Service Called | Feature Required |
|-----------|----------------|------------------|
| `position` | `set_valve_position` | SET_POSITION (4) |

---

### lawn_mower

**Description**: Robotic lawn mowers

#### Action Mappings

| Action | Service Called | Notes |
|--------|----------------|-------|
| `turn_on` | `start_mowing` | Starts mowing |
| `turn_off` | `dock` | Returns to dock |
| `toggle` | `toggle` | |

---

### alarm_control_panel

**Description**: Security alarm systems

#### Action Mappings

| Action | Service Called | Notes |
|--------|----------------|-------|
| `turn_on` | `alarm_arm_home` | Defaults to arm_home |
| `turn_off` | `alarm_disarm` | |

> **Note**: For specific arming modes, use the explicit services: `alarm_arm_home`, `alarm_arm_away`, `alarm_arm_night`

---

### scene

**Description**: Scene activation

#### Action Mappings

| Action | Service Called |
|--------|----------------|
| `turn_on` | `turn_on` |

> **Note**: Scenes only support activation (turn_on).

---

### script

**Description**: Home Assistant scripts

#### Action Mappings

| Action | Service Called |
|--------|----------------|
| `turn_on` | `turn_on` |
| `turn_off` | `turn_off` |
| `toggle` | `toggle` |

---

### automation

**Description**: Home Assistant automations

#### Action Mappings

| Action | Service Called |
|--------|----------------|
| `turn_on` | `turn_on` |
| `turn_off` | `turn_off` |
| `toggle` | `toggle` |

---

### button

**Description**: Button entities (stateless)

#### Action Mappings

| Action | Service Called |
|--------|----------------|
| `turn_on` | `press` |

---

### siren

**Description**: Siren and alarm sound devices

#### Action Mappings

| Action | Service Called |
|--------|----------------|
| `turn_on` | `turn_on` |
| `turn_off` | `turn_off` |
| `toggle` | `toggle` |

---

### camera

**Description**: Camera devices

#### Action Mappings

| Action | Service Called |
|--------|----------------|
| `turn_on` | `turn_on` |
| `turn_off` | `turn_off` |

---

### group

**Description**: Entity groups (aggregates multiple entities)

#### Action Mappings

| Action | Service Called |
|--------|----------------|
| `turn_on` | `turn_on` |
| `turn_off` | `turn_off` |
| `toggle` | `toggle` |

---

### Input Helpers

#### input_boolean

| Action | Service Called |
|--------|----------------|
| `turn_on` | `turn_on` |
| `turn_off` | `turn_off` |
| `toggle` | `toggle` |

#### input_number

| Parameter | Service Called |
|-----------|----------------|
| `value` | `set_value` |

#### input_select

| Parameter | Service Called |
|-----------|----------------|
| `option` | `select_option` |

#### input_text

| Parameter | Service Called |
|-----------|----------------|
| `value` | `set_value` |

#### input_datetime

| Action | Service Called |
|--------|----------------|
| `set_value` | `set_datetime` |

#### number

| Parameter | Service Called |
|-----------|----------------|
| `value` | `set_value` |

#### select

| Parameter | Service Called |
|-----------|----------------|
| `option` | `select_option` |

#### text

| Parameter | Service Called |
|-----------|----------------|
| `value` | `set_value` |

---

## Parameter Normalization

The tool automatically normalizes attribute names to service parameter names:

| Domain | Input (Attribute Name) | Output (Parameter Name) |
|--------|------------------------|-------------------------|
| cover | `current_position` | `position` |
| cover | `current_tilt_position` | `tilt_position` |
| climate | `current_temperature` | `temperature` |

---

## Auto-Conversions

### Light Brightness

- **Input**: `brightness_pct` (0-100 percentage)
- **Output**: `brightness` (0-255 Home Assistant native)
- **Formula**: `brightness = int(brightness_pct * 255 / 100)`

**Example**:
```json
// Input
{"action": "set_value", "entity_id": "light.living_room", "parameters": {"brightness_pct": 50}}

// Converted to
{"brightness": 127}
```

---

## Special Auto-Injection Behavior

### Climate HVAC Mode

When controlling climate entities, the `hvac_mode` parameter is automatically injected if not provided:

| Action | Auto-Injected Parameter |
|--------|-------------------------|
| `turn_off` | `hvac_mode: "off"` |
| `turn_on` | Best available mode (see priority below) |

**Turn On Mode Selection Priority**:
1. `heat_cool` (preferred for dual-mode systems)
2. `auto`
3. `heat`
4. `cool`
5. First available non-"off" mode

**Example**:
```json
// Input (no hvac_mode specified)
{"action": "turn_on", "entity_id": "climate.living_room"}

// Auto-injected (if entity supports heat_cool)
{"hvac_mode": "heat_cool"}
```

---

## Feature Validation

Certain services require specific entity features. The tool validates these before execution:

### Cover Domain

| Service | Required Feature | Feature Flag |
|---------|------------------|--------------|
| `set_cover_position` | SET_POSITION | 4 |
| `set_cover_tilt_position` | SET_TILT_POSITION | 128 |

**Error Handling**: If a cover doesn't support position control (e.g., binary garage doors), the tool returns a helpful error suggesting to use `turn_on`/`turn_off` instead.

### Valve Domain

| Service | Required Feature | Feature Flag |
|---------|------------------|--------------|
| `set_valve_position` | SET_POSITION | 4 |

---

## Fallback Behavior

For domains not explicitly mapped in `DOMAIN_SERVICE_MAPPINGS`, the tool falls back to generic service mapping:

| Action | Fallback Service |
|--------|------------------|
| `turn_on` | `turn_on` |
| `turn_off` | `turn_off` |
| `toggle` | `toggle` |
| `set_value` | `turn_on` (best guess) |

---

## File References

- **Domain Mappings**: `custom_components/home_agent/const.py` (DOMAIN_SERVICE_MAPPINGS)
- **Parameter Handling**: `custom_components/home_agent/tools/ha_control.py` (HomeAssistantControlTool)

---

*Last Updated: Based on home-agent version 0.8.4*
