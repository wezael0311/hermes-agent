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


def assert_audited_query_content(agent, request: dict) -> None:
    """Check the final provider-visible user text, after request conversion."""
    if not getattr(agent, "audited_query_transport", False):
        return
    expected = getattr(agent, "_audited_query", None)
    if not isinstance(expected, str) or not expected:
        raise RuntimeError("audited_query_content_mismatch")
    extra = request.get("extra_body")
    if isinstance(extra, dict) and any(key in extra for key in ("input", "messages")):
        raise RuntimeError("audited_query_content_mismatch")
    key = "input" if getattr(agent, "api_mode", None) == "codex_responses" else "messages"
    items = request.get(key)
    if not isinstance(items, list):
        raise RuntimeError("audited_query_content_mismatch")
    user_texts = []
    for item in items:
        if not isinstance(item, dict):
            raise RuntimeError("audited_query_content_mismatch")
        role = item.get("role")
        if role in ("system", "developer"):
            continue
        if role != "user":
            raise RuntimeError("audited_query_content_mismatch")
        content = item.get("content")
        if isinstance(content, list) and len(content) == 1:
            part = content[0]
            if isinstance(part, dict) and part.get("type") in ("input_text", "text"):
                content = part.get("text")
        if not isinstance(content, str):
            raise RuntimeError("audited_query_content_mismatch")
        user_texts.append(content)
    if user_texts != [expected]:
        raise RuntimeError("audited_query_content_mismatch")
