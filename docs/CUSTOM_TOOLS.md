# Custom Tools Guide

This guide explains how to create and configure custom tools for Home Agent, allowing you to extend the LLM's capabilities with external REST APIs and Home Assistant services.

## Table of Contents

- [Overview](#overview)
- [Configuration Location](#configuration-location)
- [Tool Types](#tool-types)
  - [REST Handler](#rest-handler)
  - [Service Handler](#service-handler)
- [Tool Schema](#tool-schema)
- [REST Handler Guide](#rest-handler-guide)
- [Service Handler Guide](#service-handler-guide)
- [Template Usage](#template-usage)
- [Response Format](#response-format)
- [Error Handling](#error-handling)
- [Security Best Practices](#security-best-practices)
- [Examples and Recipes](#examples-and-recipes)
- [Troubleshooting](#troubleshooting)

## Overview

Custom tools allow the LLM to interact with external systems and Home Assistant services through a flexible configuration-based framework. You can create tools that:

- Call external REST APIs (weather, task management, IoT platforms, etc.)
- Trigger Home Assistant automations
- Execute Home Assistant scripts
- Activate scenes
- Call any Home Assistant service

All custom tools are defined in your `configuration.yaml` file and are automatically registered when Home Agent starts.

## Configuration Location

Custom tools are configured in your Home Assistant `configuration.yaml` file under the `home_agent` integration:

```yaml
home_agent:
  tools_custom:
    - name: my_custom_tool
      description: "Description of what this tool does"
      parameters:
        # JSON Schema defining the tool's parameters
      handler:
        type: rest  # or service
        # Handler-specific configuration
```

## Tool Types

Home Agent supports two types of custom tool handlers:

### REST Handler

Calls external HTTP APIs with configurable methods, headers, query parameters, and request bodies.

**Use cases:**
- External weather APIs
- Task management systems
- IoT platform APIs
- Third-party services
- Custom web applications

### Service Handler

Calls Home Assistant services to control devices, trigger automations, run scripts, or activate scenes.

**Use cases:**
- Trigger complex automations
- Run multi-step scripts
- Activate predefined scenes
- Call specialized Home Assistant services
- Chain multiple actions together

## Tool Schema

Every custom tool must have the following fields:

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Unique identifier for the tool (lowercase, underscore-separated) |
| `description` | Yes | Clear description of what the tool does (helps LLM decide when to use it) |
| `parameters` | No | JSON Schema defining the tool's input parameters (defaults to empty object) |
| `handler` | Yes | Handler configuration (type + handler-specific settings) |

### Parameter Schema

Parameters are defined using [JSON Schema](https://json-schema.org/):

```yaml
parameters:
  type: object
  properties:
    location:
      type: string
      description: "City name or coordinates"
    units:
      type: string
      enum: [celsius, fahrenheit]
      description: "Temperature units"
  required:
    - location
```

**Supported types:**
- `string` - Text values
- `number` - Numeric values (int or float)
- `integer` - Integer values only
- `boolean` - true/false values
- `array` - Lists of values
- `object` - Nested objects

## REST Handler Guide

The REST handler enables calling external HTTP APIs with full control over the request.

### Configuration Options

```yaml
handler:
  type: rest
  url: "https://api.example.com/endpoint"  # Required
  method: GET  # Required: GET, POST, PUT, DELETE
  headers:  # Optional
    Authorization: "Bearer {{ secrets.api_key }}"
    Content-Type: "application/json"
  query_params:  # Optional (for GET requests)
    location: "{{ location }}"
    format: "json"
  body:  # Optional (for POST/PUT requests)
    title: "{{ title }}"
    status: "active"
  timeout: 30  # Optional (seconds, defaults to configured tool timeout)
```

### HTTP Methods

| Method | Use Case | Supports Body | Supports Query Params |
|--------|----------|---------------|----------------------|
| GET | Retrieve data | No | Yes |
| POST | Create resources | Yes | Yes |
| PUT | Update resources | Yes | Yes |
| DELETE | Delete resources | No | Yes |

### Headers

Headers support template rendering for dynamic values:

```yaml
headers:
  Authorization: "Bearer {{ secrets.weather_api_key }}"
  X-Custom-Header: "{{ user_id }}"
  Content-Type: "application/json"
```

### Query Parameters

Query parameters are appended to the URL:

```yaml
query_params:
  location: "{{ location }}"
  units: "metric"
  lang: "en"
```

Example URL: `https://api.example.com/forecast?location=Seattle&units=metric&lang=en`

### Request Body

Request body is sent as JSON for POST/PUT requests:

```yaml
body:
  title: "{{ task_title }}"
  description: "{{ task_description }}"
  priority: "high"
  created_by: "home_assistant"
```

### Complete REST Example

```yaml
- name: check_weather
  description: "Get current weather forecast for a location"
  parameters:
    type: object
    properties:
      location:
        type: string
        description: "City name (e.g., 'Seattle' or 'London, UK')"
      units:
        type: string
        enum: [metric, imperial]
        description: "Temperature units"
    required:
      - location
  handler:
    type: rest
    url: "https://api.openweathermap.org/data/2.5/weather"
    method: GET
    headers:
      Accept: "application/json"
    query_params:
      q: "{{ location }}"
      units: "{{ units }}"
      appid: "{{ secrets.openweather_api_key }}"
```

## Service Handler Guide

The Service handler enables calling Home Assistant services with templated parameters.

### Configuration Options

```yaml
handler:
  type: service
  service: domain.service_name  # Required (e.g., automation.trigger)
  data:  # Optional service data
    entity_id: automation.morning_routine
    skip_condition: false
  target:  # Optional target selector
    entity_id: scene.movie_time
    # or device_id, area_id
```

### Service Format

Services must be specified in `domain.service_name` format:

- `automation.trigger` - Trigger an automation
- `script.run_script` - Run a script
- `scene.turn_on` - Activate a scene
- `notify.mobile_app` - Send notification
- `climate.set_temperature` - Set thermostat

### Service Data

Service data parameters are passed to the service:

```yaml
data:
  entity_id: automation.morning_routine
  message: "{{ notification_message }}"
  temperature: "{{ target_temp }}"
```

### Target Selector

The target field specifies which entities, devices, or areas the service should affect:

```yaml
target:
  entity_id: light.living_room  # Single entity
  # or
  entity_id:  # Multiple entities
    - light.living_room
    - light.kitchen
  # or
  device_id: abc123  # Specific device
  # or
  area_id: living_room  # All entities in area
```

### Complete Service Examples

#### Trigger Automation

```yaml
- name: trigger_morning_routine
  description: "Activate the morning routine automation"
  handler:
    type: service
    service: automation.trigger
    data:
      entity_id: automation.morning_routine
      skip_condition: true
```

#### Run Script with Parameters

```yaml
- name: notify_family
  description: "Send a notification to the family with a custom message"
  parameters:
    type: object
    properties:
      message:
        type: string
        description: "The notification message to send"
      priority:
        type: string
        enum: [low, normal, high]
        description: "Notification priority level"
    required:
      - message
  handler:
    type: service
    service: script.notify_family
    data:
      message: "{{ message }}"
      priority: "{{ priority }}"
```

#### Activate Scene

```yaml
- name: set_movie_scene
  description: "Activate movie watching mode (dims lights, closes blinds)"
  handler:
    type: service
    service: scene.turn_on
    target:
      entity_id: scene.movie_time
```

#### Control Room Lights

```yaml
- name: control_room_lights
  description: "Turn lights on or off in a specific room"
  parameters:
    type: object
    properties:
      room:
        type: string
        description: "Room name (e.g., 'living_room', 'bedroom')"
      action:
        type: string
        enum: [turn_on, turn_off]
        description: "Action to perform"
    required:
      - room
      - action
  handler:
    type: service
    service: "light.{{ action }}"
    target:
      area_id: "{{ room }}"
```

## Template Usage

Both REST and Service handlers support Jinja2 template rendering for dynamic values.

### Accessing Tool Parameters

Tool parameters are available as template variables:

```yaml
# Parameter definition
parameters:
  properties:
    location:
      type: string

# Usage in template
url: "https://api.example.com/{{ location }}"
```

### Accessing Secrets

Access secrets from `secrets.yaml` using the `secrets` variable:

```yaml
headers:
  Authorization: "Bearer {{ secrets.api_key }}"
```

**secrets.yaml:**
```yaml
api_key: your_secret_api_key_here
```

### Home Assistant Templates

You can use Home Assistant template functions:

```yaml
data:
  timestamp: "{{ now().isoformat() }}"
  temperature: "{{ states('sensor.temperature') }}"
  is_home: "{{ is_state('person.john', 'home') }}"
```

### Conditional Templates

```yaml
body:
  priority: "{{ 'high' if urgent else 'normal' }}"
  status: "{{ 'active' if enabled else 'inactive' }}"
```

## Response Format

All custom tools return a standardized response format:

### Success Response

```json
{
  "success": true,
  "result": {
    // API response data or service confirmation
  },
  "error": null
}
```

### Error Response

```json
{
  "success": false,
  "result": null,
  "error": "Error description message"
}
```

### REST Response Parsing

- **JSON responses** are automatically parsed and returned in the `result` field
- **Non-JSON responses** are returned as text strings
- Content-Type header is checked to determine response type

### Service Response

Service tools return a success message:

```json
{
  "success": true,
  "result": "Service automation.trigger called successfully",
  "error": null
}
```

## Error Handling

Custom tools handle errors gracefully and return them to the LLM for appropriate user communication.

### REST Error Types

| Error Type | Description | Example |
|------------|-------------|---------|
| HTTP 4xx | Client errors (bad request, unauthorized, not found) | Invalid API key, missing parameters |
| HTTP 5xx | Server errors (service unavailable, internal error) | API down, server overload |
| Timeout | Request exceeded timeout limit | Slow API, network issues |
| Network | Connection failures | DNS failure, no internet |
| Template | Template rendering failed | Invalid template syntax |

### Service Error Types

| Error Type | Description | Example |
|------------|-------------|---------|
| ServiceNotFound | Service doesn't exist in Home Assistant | Typo in service name |
| Template | Template rendering failed | Invalid template syntax |
| Validation | Service data validation failed | Invalid entity_id format |

### Error Propagation

Errors are returned to the LLM, which can:
- Inform the user about the error
- Suggest alternatives
- Retry with different parameters
- Request clarification

## Security Best Practices

### 1. Use Secrets for API Keys

**Never hardcode API keys in configuration.yaml!**

❌ **Bad:**
```yaml
headers:
  Authorization: "Bearer sk-abc123xyz"
```

✅ **Good:**
```yaml
headers:
  Authorization: "Bearer {{ secrets.api_key }}"
```

### 2. Validate Input Parameters

Use JSON Schema to validate and constrain inputs:

```yaml
parameters:
  properties:
    email:
      type: string
      format: email  # Validates email format
    age:
      type: integer
      minimum: 0
      maximum: 120
    priority:
      type: string
      enum: [low, medium, high]  # Only allow specific values
```

### 3. Use HTTPS for REST APIs

Always use HTTPS URLs to ensure encrypted communication:

```yaml
url: "https://api.example.com/endpoint"  # ✅ Secure
# url: "http://api.example.com/endpoint"  # ❌ Insecure
```

### 4. Limit Tool Scope

Create focused tools with specific purposes rather than generic tools that do many things.

### 5. Review Service Permissions

Be mindful of which Home Assistant services you expose through custom tools. Some services can have significant impact.

### 6. Set Appropriate Timeouts

Configure reasonable timeouts to prevent hanging requests:

```yaml
handler:
  type: rest
  timeout: 10  # Seconds
```

### 7. Monitor Tool Usage

Use Home Assistant events to monitor custom tool execution:

```yaml
automation:
  - alias: "Log Custom Tool Usage"
    trigger:
      - platform: event
        event_type: home_agent.tool.executed
        event_data:
          tool_name: check_weather
    action:
      - service: notify.admin
        data:
          message: "Weather API called at {{ now() }}"
```

## Examples and Recipes

### Weather API Integration

```yaml
- name: get_weather_forecast
  description: "Get 3-day weather forecast for a location"
  parameters:
    type: object
    properties:
      city:
        type: string
        description: "City name"
      days:
        type: integer
        minimum: 1
        maximum: 7
        description: "Number of days to forecast"
    required:
      - city
  handler:
    type: rest
    url: "https://api.weatherapi.com/v1/forecast.json"
    method: GET
    query_params:
      key: "{{ secrets.weather_api_key }}"
      q: "{{ city }}"
      days: "{{ days }}"
```

### Task Management

```yaml
- name: create_task
  description: "Create a new task in Todoist"
  parameters:
    type: object
    properties:
      title:
        type: string
        description: "Task title"
      priority:
        type: integer
        minimum: 1
        maximum: 4
        description: "Priority (1=normal, 4=urgent)"
      due_date:
        type: string
        description: "Due date (YYYY-MM-DD)"
    required:
      - title
  handler:
    type: rest
    url: "https://api.todoist.com/rest/v2/tasks"
    method: POST
    headers:
      Authorization: "Bearer {{ secrets.todoist_api_key }}"
      Content-Type: "application/json"
    body:
      content: "{{ title }}"
      priority: "{{ priority }}"
      due_string: "{{ due_date }}"
```

### Smart Home Scenes

```yaml
- name: activate_scene
  description: "Activate a predefined smart home scene"
  parameters:
    type: object
    properties:
      scene_name:
        type: string
        enum: [morning, evening, movie, party, sleep]
        description: "Scene to activate"
    required:
      - scene_name
  handler:
    type: service
    service: scene.turn_on
    target:
      entity_id: "scene.{{ scene_name }}"
```

### Climate Control

```yaml
- name: set_room_temperature
  description: "Set temperature for a specific room's thermostat"
  parameters:
    type: object
    properties:
      room:
        type: string
        enum: [living_room, bedroom, office]
        description: "Room name"
      temperature:
        type: number
        minimum: 15
        maximum: 30
        description: "Target temperature in Celsius"
    required:
      - room
      - temperature
  handler:
    type: service
    service: climate.set_temperature
    target:
      entity_id: "climate.{{ room }}"
    data:
      temperature: "{{ temperature }}"
```

### Notification System

```yaml
- name: send_notification
  description: "Send a notification to family members"
  parameters:
    type: object
    properties:
      message:
        type: string
        description: "Notification message"
      who:
        type: string
        enum: [everyone, john, jane, kids]
        description: "Who should receive the notification"
      urgent:
        type: boolean
        description: "Mark as urgent (critical priority)"
    required:
      - message
      - who
  handler:
    type: service
    service: notify.family
    data:
      message: "{{ message }}"
      title: "Home Assistant"
      data:
        priority: "{{ 'high' if urgent else 'normal' }}"
        tag: "home_agent"
        group: "{{ who }}"
```

## Troubleshooting

### Tool Not Appearing in LLM Tools List

**Symptoms:** Custom tool doesn't show up when checking tool definitions

**Solutions:**
1. Check YAML syntax in `configuration.yaml`
2. Verify all required fields are present (`name`, `description`, `handler`)
3. Restart Home Assistant after configuration changes
4. Check Home Assistant logs for validation errors

### Template Rendering Errors

**Symptoms:** Error message about template rendering failure

**Solutions:**
1. Verify template syntax (correct Jinja2 format: `{{ variable }}`)
2. Ensure variable names match parameter names exactly
3. Check that secrets are defined in `secrets.yaml`
4. Test templates in Developer Tools > Template

### Service Not Found Errors

**Symptoms:** `ServiceNotFound` error when executing service tool

**Solutions:**
1. Verify service exists: Developer Tools > Services
2. Check service name format: `domain.service_name`
3. Ensure integration providing service is loaded
4. Check entity_id exists if targeting specific entity

### REST API Errors

**Symptoms:** HTTP errors (401, 404, 500, etc.)

**Solutions:**
1. **401 Unauthorized**: Check API key is correct and not expired
2. **404 Not Found**: Verify URL is correct and endpoint exists
3. **500 Server Error**: API may be down, check API status page
4. **Timeout**: Increase timeout value or check network connectivity

### Tool Executes But Returns Unexpected Results

**Symptoms:** Tool completes successfully but result is not as expected

**Solutions:**
1. Check parameter values are being passed correctly
2. Verify template rendering is producing expected values
3. Test API directly (curl, Postman) with same parameters
4. Enable debug logging to see full request/response
5. Check API documentation for required/optional parameters

### Debug Logging

Enable debug logging for detailed troubleshooting:

```yaml
logger:
  default: info
  logs:
    custom_components.home_agent: debug
    custom_components.home_agent.tools.custom: debug
```

### Testing Tools Manually

Test custom tools using the `home_agent.execute_tool` service:

```yaml
service: home_agent.execute_tool
data:
  tool_name: check_weather
  parameters:
    location: "Seattle"
```

### Common Configuration Mistakes

1. **Missing quotes around template variables**
   ```yaml
   # ❌ Wrong
   location: {{ city }}

   # ✅ Correct
   location: "{{ city }}"
   ```

2. **Incorrect service format**
   ```yaml
   # ❌ Wrong
   service: trigger automation

   # ✅ Correct
   service: automation.trigger
   ```

3. **Parameters not defined in schema**
   ```yaml
   # Tool uses {{ priority }} but parameter not defined
   parameters:
     properties:
       # priority is missing!
       title:
         type: string
   ```

4. **Hardcoded secrets**
   ```yaml
   # ❌ Never do this
   headers:
     Authorization: "Bearer sk-abc123"

   # ✅ Always use secrets
   headers:
     Authorization: "Bearer {{ secrets.api_key }}"
   ```

## Further Reading

- [Home Assistant Services Documentation](https://www.home-assistant.io/docs/scripts/service-calls/)
- [JSON Schema Documentation](https://json-schema.org/understanding-json-schema/)
- [Jinja2 Template Documentation](https://jinja.palletsprojects.com/)
- [Home Assistant Templating](https://www.home-assistant.io/docs/configuration/templating/)
- [External LLM Guide](EXTERNAL_LLM.md)
- [Project Specification](PROJECT_SPEC.md)
