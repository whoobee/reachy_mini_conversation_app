"""Home Assistant administration through ha-mcp, with a time-limited "engineering mode" for writes.

ha-mcp (https://github.com/homeassistant-ai/ha-mcp) exposes ~80 tools for automations, scripts,
helpers, blueprints, logs, traces and registries. Their definitions are tens of thousands of
tokens, too much to send with every voice turn, so the voice agent gets four small gateway tools:
search tools, describe one, run one, and toggle engineering mode.

Read tools (those listed on ha-mcp's ``/readonly`` endpoint) always run, through that endpoint.
Write tools run only while engineering mode is on, through the full endpoint. Engineering mode
switches itself off after a timeout.
"""

from __future__ import annotations
import re
import json
import time
import asyncio
import logging
from typing import Any, Final
from collections.abc import Callable


logger = logging.getLogger(__name__)

TOOL_LIST_TTL_S: Final[float] = 3600.0
# Everyday device control that needs no engineering mode (unless the entity is protected).
SAFE_SERVICE_DOMAINS: Final[frozenset[str]] = frozenset(
    {"light", "switch", "cover", "fan", "climate", "media_player", "scene", "humidifier", "input_boolean"}
)
DEFAULT_PROTECTED_PATTERN: Final[str] = r"valve|lock|alarm|garage|siren|heater|pump"
RESULT_CHARS: Final[int] = 8000
SEARCH_RESULTS: Final[int] = 8


class EngineeringMode:
    """Permission for Home Assistant changes that ends after a period of inactivity.

    The timeout restarts on every ``touch`` (each user utterance and admin tool call), so the
    mode stays on while someone is actively working and switches off once they stop.
    """

    def __init__(self, default_minutes: float, clock: Callable[[], float] = time.monotonic) -> None:
        """Start disabled; ``default_minutes`` is used when enabling without a duration."""
        self.default_minutes = default_minutes
        self._clock = clock
        self._until = 0.0
        self._timeout_s = 60 * default_minutes

    @property
    def active(self) -> bool:
        """Return whether engineering mode is on."""
        return self._clock() < self._until

    def seconds_left(self) -> int:
        """Return the remaining time in seconds (0 when off)."""
        return max(0, round(self._until - self._clock()))

    def enable(self, minutes: float | None = None) -> None:
        """Turn on with an inactivity timeout of ``minutes`` (default timeout when None)."""
        self._timeout_s = 60 * (minutes if minutes and minutes > 0 else self.default_minutes)
        self._until = self._clock() + self._timeout_s

    def touch(self) -> None:
        """Restart the inactivity timeout if the mode is on (never turns it on)."""
        if self.active:
            self._until = self._clock() + self._timeout_s

    def disable(self) -> None:
        """Turn off now."""
        self._until = 0.0

    def expired_since_check(self) -> bool:
        """Return True once when the mode has just timed out (for announcing it)."""
        if self._until and not self.active:
            self._until = 0.0
            return True
        return False


class HaAdmin:
    """Gateway to an ha-mcp server: tool search, description, gated execution."""

    def __init__(self, url: str, mode: EngineeringMode, protected_pattern: str | None = None) -> None:
        """``url`` is ha-mcp's full Streamable HTTP endpoint (the read-only one is ``url + '/readonly'``).

        ``protected_pattern`` is a regex; matching entity IDs need engineering mode even in safe domains.
        """
        self._url = url.rstrip("/")
        self.mode = mode
        self._protected = re.compile(protected_pattern or DEFAULT_PROTECTED_PATTERN, re.IGNORECASE)
        self._tools: dict[str, dict[str, Any]] = {}
        self._read_only: set[str] = set()
        self._listed_at = 0.0
        self._sessions: dict[str, tuple[Any, Any]] = {}
        self._lock = asyncio.Lock()

    async def _session(self, url: str) -> Any:
        """Return a connected MCP session for ``url``, opening it on first use."""
        from contextlib import AsyncExitStack

        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        if url not in self._sessions:
            stack = AsyncExitStack()
            try:
                read, write, _ = await stack.enter_async_context(streamablehttp_client(url))
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
            except BaseException:
                await stack.aclose()
                raise
            self._sessions[url] = (stack, session)
        return self._sessions[url][1]

    async def _drop_session(self, url: str) -> None:
        stack_and_session = self._sessions.pop(url, None)
        if stack_and_session is not None:
            try:
                await stack_and_session[0].aclose()
            except BaseException as e:  # a broken transport may fail to close cleanly
                logger.debug("Closing ha-mcp session failed: %s", e)

    async def _session_call(self, url: str, method: str, *args: Any) -> Any:
        """Call ``method`` on a persistent session, reconnecting once if the connection dropped."""
        async with self._lock:
            for attempt in (1, 2):
                try:
                    session = await self._session(url)
                    return await getattr(session, method)(*args)
                except Exception:
                    await self._drop_session(url)
                    if attempt == 2:
                        raise

    async def close(self) -> None:
        """Close the persistent sessions."""
        for url in list(self._sessions):
            await self._drop_session(url)

    async def _ensure_tools(self) -> None:
        if self._tools and time.monotonic() - self._listed_at < TOOL_LIST_TTL_S:
            return
        full = await self._session_call(self._url, "list_tools")
        read_only = await self._session_call(f"{self._url}/readonly", "list_tools")
        self._tools = {
            t.name: {"name": t.name, "description": t.description or "", "inputSchema": t.inputSchema}
            for t in full.tools
        }
        self._read_only = {t.name for t in read_only.tools}
        self._listed_at = time.monotonic()
        logger.info("ha-mcp tools: %d (%d read-only)", len(self._tools), len(self._read_only))

    def _kind(self, name: str) -> str:
        return "read" if name in self._read_only else "write"

    # ---- tool handlers ----

    async def find_tools(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Search ha-mcp tools by keywords in their names and descriptions."""
        self.mode.touch()
        await self._ensure_tools()
        words = [w for w in re.findall(r"\w+", str(arguments.get("query") or "").lower()) if len(w) > 2]
        scored = []
        for tool in self._tools.values():
            text = f"{tool['name']} {tool['description']}".lower()
            score = sum(3 if w in tool["name"].lower() else 1 for w in words if w in text)
            if score or not words:
                scored.append((score, tool))
        scored.sort(key=lambda item: -item[0])
        return {
            "engineering_mode": self.mode.active,
            "tools": [
                {
                    "name": tool["name"],
                    "kind": self._kind(tool["name"]),
                    "summary": tool["description"].split("\n")[0][:200],
                }
                for _, tool in scored[:SEARCH_RESULTS]
            ],
        }

    async def describe_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Return one tool's description and input schema."""
        self.mode.touch()
        await self._ensure_tools()
        tool = self._tools.get(str(arguments.get("name") or ""))
        if tool is None:
            return {"error": "unknown tool; use ha_find_tools first"}
        return {**tool, "description": tool["description"][:4000], "kind": self._kind(tool["name"])}

    async def run_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Run a tool: reads always (read-only endpoint), writes only in engineering mode."""
        self.mode.touch()
        await self._ensure_tools()
        name = str(arguments.get("name") or "")
        if name not in self._tools:
            return {"error": "unknown tool; use ha_find_tools first"}
        tool_arguments = arguments.get("arguments") or {}
        if isinstance(tool_arguments, str):
            try:
                tool_arguments = json.loads(tool_arguments)
            except ValueError:
                return {"error": "arguments must be a JSON object"}
        kind = self._kind(name)
        if kind == "write" and not self.mode.active:
            return {
                "error": "engineering mode is off: this tool changes Home Assistant. Ask the user to turn "
                "engineering mode on (set_engineering_mode) and confirm the change first."
            }
        url = self._url if kind == "write" else f"{self._url}/readonly"
        logger.info("ha-mcp %s tool %s %s", kind, name, json.dumps(tool_arguments, ensure_ascii=False)[:300])
        try:
            result = await self._session_call(url, "call_tool", name, tool_arguments)
        except Exception as e:
            return {"error": f"ha-mcp call failed: {type(e).__name__}: {e}"}
        text = "\n".join(getattr(block, "text", "") for block in result.content if getattr(block, "text", ""))
        response: dict[str, Any] = {"tool": name, "kind": kind, "is_error": bool(result.isError)}
        response["result"] = text[:RESULT_CHARS] + (" …(truncated)" if len(text) > RESULT_CHARS else "")
        return response

    def service_blocked_reason(self, domain: str, service: str, entity_ids: list[str]) -> str | None:
        """Return why a service call needs engineering mode, or None when it may run now."""
        if self.mode.active:
            return None
        if domain not in SAFE_SERVICE_DOMAINS:
            return f"{domain}.{service} changes Home Assistant itself or a sensitive system"
        if not entity_ids:
            return "a service call without entity_id could affect many devices"
        protected = [e for e in entity_ids if self._protected.search(e)]
        if protected:
            return f"{', '.join(protected)} is a protected device"
        return None

    async def call_service(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Device control: safe domains always, everything else only in engineering mode."""
        self.mode.touch()
        domain = str(arguments.get("domain") or "").strip()
        service = str(arguments.get("service") or "").strip()
        data = arguments.get("data") or {}
        if isinstance(data, str):
            try:
                data = json.loads(data) if data.strip() else {}
            except ValueError:
                return {"error": "data must be a JSON object"}
        if not isinstance(data, dict):
            return {"error": "data must be a JSON object"}
        entity = arguments.get("entity_id") or data.get("entity_id")
        entity_ids = [entity] if isinstance(entity, str) else [str(e) for e in entity or []]
        if not domain or not service:
            return {"error": "domain and service are required, e.g. light / turn_on"}
        reason = self.service_blocked_reason(domain, service, entity_ids)
        if reason:
            return {"error": f"engineering mode is off: {reason}. Ask the user to turn engineering mode on first."}
        call: dict[str, Any] = {
            "domain": domain,
            "service": service,
            "data": {k: v for k, v in data.items() if k != "entity_id"},
        }
        if entity_ids:
            call["entity_id"] = entity_ids[0] if len(entity_ids) == 1 else ",".join(entity_ids)
        logger.info("ha-mcp service %s.%s %s", domain, service, entity_ids)
        try:
            result = await self._session_call(self._url, "call_tool", "ha_call_service", call)
        except Exception as e:
            return {"error": f"ha-mcp call failed: {type(e).__name__}: {e}"}
        text = "\n".join(getattr(block, "text", "") for block in result.content if getattr(block, "text", ""))
        return {"is_error": bool(result.isError), "result": text[:RESULT_CHARS]}

    async def get_state(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Read one or more entity states."""
        return await self.run_tool({"name": "ha_get_state", "arguments": {"entity_id": arguments.get("entity_id")}})

    async def list_entities(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Find entities by text, domain or area."""
        query: dict[str, Any] = {"limit": int(arguments.get("limit") or 40)}
        for source, target in (("query", "query"), ("domain", "domain_filter"), ("area", "area_filter")):
            if arguments.get(source):
                query[target] = arguments[source]
        return await self.run_tool({"name": "ha_search", "arguments": query})

    async def engineering_mode(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Turn engineering mode on/off, or report its state when ``enabled`` is omitted."""
        enabled = arguments.get("enabled")
        if enabled is True:
            minutes = arguments.get("minutes")
            self.mode.enable(float(minutes) if minutes else None)
            logger.info("Engineering mode ON for %d s", self.mode.seconds_left())
        elif enabled is False:
            self.mode.disable()
            logger.info("Engineering mode OFF")
        return {"engineering_mode": self.mode.active, "seconds_left": self.mode.seconds_left()}


TOOL_DEFINITIONS: Final[dict[str, tuple[str, dict[str, Any]]]] = {
    "ha_find_tools": (
        "Search the Home Assistant administration tools (ha-mcp: automations, scripts, helpers, blueprints, "
        "entities, devices, areas, dashboards, logs, traces, history, system health, backups, updates). "
        "Returns tool names, whether they read or write, and a summary. Use for diagnostics or configuration, "
        "not for simply switching a light (use ha_call_service for that).",
        {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Keywords, e.g. 'automation trace', 'logs'."}},
            "required": ["query"],
        },
    ),
    "ha_describe_tool": (
        "Get the full description and parameters of one Home Assistant administration tool before running it.",
        {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    ),
    "ha_run_tool": (
        "Run a Home Assistant administration tool by name. Read tools always work; write tools (changes to "
        "automations, helpers, entities, devices, config, restarts) only work while engineering mode is on.",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Tool name from ha_find_tools."},
                "arguments": {"type": "object", "description": "Tool arguments as described by ha_describe_tool."},
            },
            "required": ["name"],
        },
    ),
    "ha_call_service": (
        "Control a Home Assistant device: lights, switches, blinds (cover), fans, climate, media players, scenes. "
        "Example: domain 'light', service 'turn_on', entity_id 'light.kitchen'. Other domains (automations, "
        "scripts, restarts, locks, alarms) and protected devices (valves, heaters, pumps...) need engineering mode.",
        {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Service domain, e.g. 'light', 'cover'."},
                "service": {"type": "string", "description": "Service, e.g. 'turn_on', 'turn_off', 'open_cover'."},
                "entity_id": {"type": "string", "description": "Target entity id."},
                "data": {"type": "object", "description": "Extra service data, e.g. {'brightness_pct': 50}."},
            },
            "required": ["domain", "service", "entity_id"],
        },
    ),
    "ha_get_state": (
        "Read the current state and attributes of a Home Assistant entity (e.g. a temperature sensor).",
        {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string", "description": "Entity id, e.g. 'sensor.bedroom_temperature'."}
            },
            "required": ["entity_id"],
        },
    ),
    "ha_list_entities": (
        "Find Home Assistant entities by text, domain (e.g. 'light') and/or area, with their current states.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Text to match, e.g. 'kitchen'."},
                "domain": {"type": "string", "description": "Entity domain filter, e.g. 'light'."},
                "area": {"type": "string", "description": "Area filter, e.g. 'living room'."},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            },
        },
    ),
    "set_engineering_mode": (
        "Turn Reachy's engineering mode on or off. While on, the Home Assistant write tools work; it switches "
        "itself off after the given minutes without activity (any request or tool use restarts the timer). "
        "Omit 'enabled' to read the current state.",
        {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean", "description": "True to turn on, false to turn off."},
                "minutes": {
                    "type": "number",
                    "minimum": 1,
                    "maximum": 240,
                    "description": "Duration when turning on.",
                },
            },
        },
    ),
}
