# MCP Server Support

Home Agent can connect to remote [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) servers over HTTP or Server-Sent Events (SSE).  This mirrors the `mcp.json` configuration used by VSCode agents but is expressed in your Home Assistant `configuration.yaml`.

**Scope:** Only `http` and `sse` transports are supported.  Locally executed `stdio` servers with command/args are **not** supported.

## Configuration

Add an `mcp_servers` list under the `home_agent:` key in `configuration.yaml`:

```yaml
home_agent:
  mcp_servers:
    - name: github
      type: http
      url: https://api.githubcopilot.com/mcp
    - name: local
      type: sse
      url: http://localhost:3000/mcp
      headers:
        Authorization: !secret mcp_token
```

### Options

| Key | Required | Description |
|-----|----------|-------------|
| `name` | Yes | Unique identifier for the server.  Tools are exposed as `<name>__<tool_name>`. |
| `type` | No | Transport type.  `http` (default) or `sse`. |
| `url` | Yes | Base URL of the MCP server.  The discovery endpoint is `<url>/tools` and the invocation endpoint is `<url>/tools/<tool_name>`. |
| `headers` | No | Static HTTP headers to include on every request, e.g. `Authorization`. |
| `timeout` | No | Request timeout in seconds.  Defaults to `30`. |

## Tool namespacing

To avoid collisions with the built-in `ha_control` and `ha_query` tools (or custom tools), every tool exposed by an MCP server is prefixed with the server name and two underscores.  For example, a server named `github` exposing a tool called `search_issues` becomes `github__search_issues`.

## Discovery and execution

When the integration loads, it queries each configured server's `<url>/tools` endpoint and registers the discovered tools.  When the LLM calls one of these tools, Home Agent POSTs the LLM-provided arguments to `<url>/tools/<tool_name>` and forwards the result back to the LLM.

## Security considerations

- Do **not** commit API keys directly in `configuration.yaml`.  Use [Home Assistant secrets](https://www.home-assistant.io/docs/configuration/secrets/) for `headers.Authorization` or other sensitive values.
- Home Agent does not run arbitrary `stdio` commands.  It only connects to HTTP or SSE servers that you explicitly configure.
