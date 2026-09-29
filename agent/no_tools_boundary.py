"""Fail-closed request boundary for explicit CLI no-tools sessions."""


def assert_no_tools_boundary(agent, request: dict) -> None:
    """Reject any tool capability before a provider request is sent."""
    if getattr(agent, "tool_policy", None) != "no_tools":
        return
    if (
        getattr(agent, "enabled_toolsets", None) != []
        or getattr(agent, "tools", None) != []
        or getattr(agent, "valid_tool_names", set())
    ):
        raise RuntimeError("no_tools_boundary_violation")
    if request.get("tools") or request.get("functions"):
        raise RuntimeError("no_tools_boundary_violation")
    if request.get("tool_choice") not in (None, "none"):
        raise RuntimeError("no_tools_boundary_violation")
    if request.get("parallel_tool_calls"):
        raise RuntimeError("no_tools_boundary_violation")
    extra = request.get("extra_body")
    if isinstance(extra, dict) and any(
        extra.get(key) for key in ("tools", "functions", "tool_choice", "parallel_tool_calls")
    ):
        raise RuntimeError("no_tools_boundary_violation")
