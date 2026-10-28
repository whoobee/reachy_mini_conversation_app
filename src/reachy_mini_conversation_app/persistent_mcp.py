"""A persistent MCP client session that survives server restarts.

The MCP SDK's streamable HTTP client must be entered and exited in the same task, and when the
server goes away it cancels the task that opened it. Opening it inside a request handler and
closing it from another one therefore crashes on reconnect. Here a background task owns each
connection; callers only use the session, and a dead connection is replaced on the next call.
"""

from __future__ import annotations
import asyncio
import logging
import datetime as dt
from typing import Any


logger = logging.getLogger(__name__)


def _not_delivered(error: BaseException) -> bool:
    """Whether ``error`` means the request never reached the server (safe to retry)."""
    import httpx

    seen: set[int] = set()
    stack: list[BaseException] = [error]
    while stack:
        e = stack.pop()
        if id(e) in seen:
            continue
        seen.add(id(e))
        if isinstance(e, (httpx.ConnectError, ConnectionRefusedError)):
            return True
        stack.extend(getattr(e, "exceptions", ()))
        stack.extend(x for x in (e.__cause__, e.__context__) if x is not None)
    return False


class PersistentMcpClient:
    """One MCP client session, owned by a background task and reopened when it dies."""

    def __init__(self, url: str, headers: dict[str, str] | None = None, timeout_s: float = 60.0) -> None:
        """Connect to the streamable HTTP MCP endpoint ``url`` on first use."""
        self._url = url
        self._headers = headers or {}
        self._timeout_s = timeout_s
        self._session: Any = None
        self._owner: asyncio.Task[BaseException | None] | None = None
        self._closing: asyncio.Event | None = None
        self._lock = asyncio.Lock()

    async def _own(self, ready: asyncio.Future[Any], closing: asyncio.Event) -> BaseException | None:
        """Hold one connection open until ``closing`` is set; return why it ended, if it failed."""
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        try:
            async with streamablehttp_client(self._url, headers=self._headers, timeout=self._timeout_s) as (r, w, _):
                timeout = dt.timedelta(seconds=self._timeout_s)
                async with ClientSession(r, w, read_timeout_seconds=timeout) as session:
                    await session.initialize()
                    ready.set_result(session)
                    await closing.wait()
        except BaseException as e:  # includes the SDK cancelling us when the server disappears
            if not ready.done():
                ready.set_exception(e if isinstance(e, Exception) else ConnectionError(str(e) or type(e).__name__))
            else:
                logger.debug("MCP connection to %s ended: %s", self._url, type(e).__name__)
            return e
        return None

    async def _get(self) -> tuple[Any, asyncio.Task[BaseException | None]]:
        async with self._lock:
            if self._session is None or self._owner is None or self._owner.done():
                ready: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
                closing = asyncio.Event()
                owner = asyncio.create_task(self._own(ready, closing), name=f"mcp-client:{self._url}")
                try:
                    session = await ready
                except BaseException:
                    closing.set()
                    raise
                self._session, self._owner, self._closing = session, owner, closing
            assert self._owner is not None
            return self._session, self._owner

    @staticmethod
    async def _call(session: Any, owner: asyncio.Task[BaseException | None], method: str, *args: Any) -> Any:
        """Run one request, failing at once if the connection dies (the SDK would wait for the timeout)."""
        call = asyncio.ensure_future(getattr(session, method)(*args))
        await asyncio.wait({call, owner}, return_when=asyncio.FIRST_COMPLETED)
        if call.done():
            return call.result()
        call.cancel()
        reason = owner.result() if not owner.cancelled() else None
        raise ConnectionError("MCP connection lost") from reason

    async def reset(self) -> None:
        """Close the current connection; the next call opens a new one."""
        owner, closing = self._owner, self._closing
        self._session = self._owner = self._closing = None
        if closing is not None:
            closing.set()
        if owner is not None and owner is not asyncio.current_task():
            try:
                await asyncio.wait_for(asyncio.shield(owner), timeout=5)
            except BaseException as e:
                logger.debug("Closing MCP connection to %s: %s", self._url, type(e).__name__)

    async def request(self, method: str, *args: Any, idempotent: bool = False) -> Any:
        """Call ``method`` on the session (e.g. "list_tools", "call_tool").

        A request that provably never reached the server is retried once on a fresh connection;
        others are retried only when ``idempotent``, so a tool call never runs twice.
        """
        for attempt in (1, 2):
            session, owner = await self._get()
            try:
                return await self._call(session, owner, method, *args)
            except Exception as e:
                await self.reset()
                if attempt == 2 or not (idempotent or _not_delivered(e)):
                    raise
        raise RuntimeError("unreachable")

    async def close(self) -> None:
        """Close the connection for good."""
        await self.reset()
