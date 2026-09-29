"""Machine-checkable audit receipt for explicit no-tools CLI turns."""

import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4


def validate_no_tools_receipt(data: dict, query: str) -> None:
    encoded = query.encode("utf-8")
    expected = {
        "receipt_schema_version": 1,
        "tool_policy": "no_tools",
        "tools_enabled": False,
        "requested_toolsets": [],
        "effective_toolsets": [],
        "effective_tool_count": 0,
        "query_sha256": hashlib.sha256(encoded).hexdigest(),
        "query_byte_count": len(encoded),
    }
    if any(data.get(key) != value for key, value in expected.items()):
        raise ValueError("no_tools_receipt_invalid")
    if data.get("tools_enabled") is not False or type(data.get("effective_tool_count")) is not int:
        raise ValueError("no_tools_receipt_invalid")
    if data.get("transport") != "hermes-cli" or data.get("transport_version") != "0.20.5":
        raise ValueError("no_tools_receipt_invalid")
    if data.get("external_send_state") not in {"not_sent", "dispatch_started"}:
        raise ValueError("no_tools_receipt_invalid")
    if not data.get("provider") or not data.get("model"):
        raise ValueError("no_tools_receipt_invalid")


def _write_receipt(agent) -> None:
    path = agent._no_tools_receipt_path
    data = agent._no_tools_receipt
    validate_no_tools_receipt(data, agent._no_tools_query)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def start_no_tools_receipt(agent, query: str) -> Path:
    if getattr(agent, "tool_policy", None) != "no_tools":
        raise ValueError("no_tools_receipt_requires_no_tools")
    encoded = query.encode("utf-8")
    home = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
    agent._no_tools_query = query
    agent._no_tools_receipt_path = home / "receipts" / f"no-tools-{uuid4().hex}.json"
    agent._no_tools_receipt = {
        "receipt_schema_version": 1,
        "transport": "hermes-cli",
        "transport_version": "0.20.5",
        "tool_policy": "no_tools",
        "tools_enabled": False,
        "requested_toolsets": [],
        "effective_toolsets": [],
        "effective_tool_count": 0,
        "query_sha256": hashlib.sha256(encoded).hexdigest(),
        "query_byte_count": len(encoded),
        "provider": str(agent.provider),
        "model": str(agent.model),
        "external_send_state": "not_sent",
    }
    _write_receipt(agent)
    return agent._no_tools_receipt_path


def mark_no_tools_dispatch(agent, request: dict) -> None:
    if getattr(agent, "tool_policy", None) != "no_tools":
        return
    if not hasattr(agent, "_no_tools_receipt"):
        raise RuntimeError("no_tools_receipt_missing")
    try:
        recorded = json.loads(agent._no_tools_receipt_path.read_text(encoding="utf-8"))
        validate_no_tools_receipt(recorded, agent._no_tools_query)
        if recorded != agent._no_tools_receipt:
            raise ValueError("no_tools_receipt_invalid")
    except (OSError, ValueError) as exc:
        raise RuntimeError("no_tools_receipt_invalid") from exc
    agent._no_tools_receipt["provider"] = str(agent.provider)
    agent._no_tools_receipt["model"] = str(request.get("model") or request.get("modelId") or agent.model)
    agent._no_tools_receipt["external_send_state"] = "dispatch_started"
    _write_receipt(agent)
