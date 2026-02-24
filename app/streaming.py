"""
VERIFAI Real-Time SSE Streaming

Server-Sent Events endpoint that streams agent progress in real-time.
The frontend connects via EventSource and receives live updates as each
agent starts, completes, or encounters errors.

Architecture:
    - Each workflow session gets an asyncio.Queue
    - Workflow nodes call emit_agent_event() to push events
    - The SSE endpoint yields events from the queue
    - Queue is cleaned up when the client disconnects or workflow ends
"""

import asyncio
import json
import threading
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

streaming_router = APIRouter(tags=["streaming"])

# ─── In-memory event bus (keyed by session_id) ───
# Thread-safe dict: workflow runs in a background thread,
# SSE endpoint runs in the async event loop.
_event_queues: dict[str, asyncio.Queue] = {}
_loop: Optional[asyncio.AbstractEventLoop] = None


def _get_loop() -> asyncio.AbstractEventLoop:
    """Get the running event loop (set on first SSE connection)."""
    global _loop
    if _loop is None:
        try:
            _loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
    return _loop


def register_session(session_id: str):
    """Create an event queue for a session. Call before starting workflow."""
    global _loop
    try:
        _loop = asyncio.get_running_loop()
    except RuntimeError:
        pass
    _event_queues[session_id] = asyncio.Queue()


def unregister_session(session_id: str):
    """Remove event queue when workflow completes."""
    _event_queues.pop(session_id, None)


def emit_agent_event(
    session_id: str,
    agent: str,
    status: str,
    data: dict = None,
    message: str = ""
):
    """
    Emit a real-time event from any agent node.
    
    Thread-safe: can be called from the background workflow thread.
    The event is pushed into the asyncio queue via call_soon_threadsafe.
    
    Args:
        session_id: Workflow session ID
        agent: Agent name (e.g., "radiologist", "critic", "debate")
        status: Event status ("started", "completed", "error", "info")
        data: Optional dict with agent-specific data
        message: Optional human-readable message
    """
    if session_id not in _event_queues:
        return  # No SSE listener connected

    event = {
        "agent": agent,
        "status": status,
        "message": message or f"{agent} {status}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": data or {}
    }

    queue = _event_queues[session_id]
    loop = _get_loop()

    if loop and loop.is_running():
        # Called from background thread — use threadsafe method
        loop.call_soon_threadsafe(queue.put_nowait, event)
    else:
        # Called from async context
        try:
            queue.put_nowait(event)
        except Exception:
            pass


# ─── SSE Endpoint ───

@streaming_router.get("/workflows/{session_id}/stream")
async def stream_workflow(session_id: str):
    """
    SSE endpoint for real-time agent progress.
    
    Connect with EventSource:
        const es = new EventSource('/api/v1/workflows/{id}/stream');
        es.onmessage = (e) => { console.log(JSON.parse(e.data)); };
    
    Events have the format:
        data: {"agent": "radiologist", "status": "completed", "message": "...", "data": {...}}
    
    The stream ends when a "workflow_complete" status is received,
    or after 10 minutes of inactivity.
    """
    # Register queue if not already present
    if session_id not in _event_queues:
        register_session(session_id)

    queue = _event_queues[session_id]

    async def event_generator():
        try:
            # Send initial connection event
            hello = {
                "agent": "system",
                "status": "connected",
                "message": f"Connected to workflow {session_id[:8]}...",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": {}
            }
            yield f"data: {json.dumps(hello)}\n\n"

            while True:
                try:
                    # Wait for next event (timeout: 10 min)
                    event = await asyncio.wait_for(queue.get(), timeout=600)
                    yield f"data: {json.dumps(event)}\n\n"

                    # End stream on workflow completion
                    if event.get("status") in ("workflow_complete", "workflow_error"):
                        break
                except asyncio.TimeoutError:
                    # Send keepalive ping
                    yield f": keepalive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            # Don't unregister here — workflow might still be emitting
            pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )
