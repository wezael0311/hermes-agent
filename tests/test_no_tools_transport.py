"""Explicit no-tools transport boundaries; no provider network calls."""

from types import SimpleNamespace

import pytest

from agent.chat_completion_helpers import (
    _dispatch_nonstreaming_api_request,
    interruptible_streaming_api_call,
)
from agent.no_tools_boundary import assert_no_tools_boundary
from agent.no_tools_receipt import (
    start_no_tools_receipt,
    validate_no_tools_receipt,
)
from hermes_cli._parser import build_top_level_parser


def _agent():
    return SimpleNamespace(
        tool_policy="no_tools", enabled_toolsets=[], tools=[],
        api_mode="chat_completions", provider="openai-codex", model="gpt-test",
    )


def test_chat_option_is_distinct_from_default_and_toolsets():
    parser = build_top_level_parser()[0]
    default = parser.parse_args(["chat", "-q", "hello"])
    explicit = parser.parse_args(["chat", "-q", "hello", "-t", "web"])
    no_tools = parser.parse_args(["chat", "-q", "hello", "--no-tools"])
    assert default.toolsets is None and not default.no_tools
    assert explicit.toolsets == "web" and not explicit.no_tools
    assert no_tools.toolsets is None and no_tools.no_tools


def test_no_tools_rejects_toolset_before_startup(capsys):
    from hermes_cli.main import _prepare_agent_startup
    parser = build_top_level_parser()[0]
    args = parser.parse_args(["chat", "--no-tools", "--toolsets", "web"])
    with pytest.raises(SystemExit) as exc:
        _prepare_agent_startup(args)
    assert exc.value.code == 2
    assert "no_tools_conflicts_with_toolsets" in capsys.readouterr().err


@pytest.mark.parametrize("injection", [
    {"tools": [{"type": "function", "function": {"name": "web_search"}}]},
    {"tool_choice": "auto"},
    {"extra_body": {"tools": [{"name": "mcp_tool"}]}},
])
def test_injected_provider_tool_is_rejected_before_client(injection, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = _agent()
    start_no_tools_receipt(agent, "hello")
    def forbidden_client(*args, **kwargs):
        pytest.fail("provider client created before no-tools boundary")
    with pytest.raises(RuntimeError, match="no_tools_boundary_violation"):
        _dispatch_nonstreaming_api_request(
            agent, {"model": "gpt-test", "messages": [], **injection},
            make_client=forbidden_client,
        )
    assert agent._no_tools_receipt["external_send_state"] == "not_sent"


def test_zero_tool_payload_and_receipt(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = _agent()
    path = start_no_tools_receipt(agent, "hello")
    seen = []
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: seen.append(kwargs) or "ok"
    )))
    result = _dispatch_nonstreaming_api_request(
        agent, {"model": "gpt-test", "messages": []},
        make_client=lambda *args, **kwargs: client,
    )
    assert result == "ok" and len(seen) == 1
    assert "tools" not in seen[0] and "tool_choice" not in seen[0]
    import json
    receipt = json.loads(path.read_text(encoding="utf-8"))
    validate_no_tools_receipt(receipt, "hello")
    assert receipt["effective_tool_count"] == 0
    assert receipt["external_send_state"] == "dispatch_started"
    for field, value in (
        ("tools_enabled", True), ("effective_toolsets", ["web"]),
        ("effective_tool_count", 1), ("query_sha256", "bad"),
    ):
        altered = {**receipt, field: value}
        with pytest.raises(ValueError, match="no_tools_receipt_invalid"):
            validate_no_tools_receipt(altered, "hello")


def test_agent_tool_injection_is_rejected():
    agent = _agent()
    agent.tools = [{"type": "function", "function": {"name": "web_search"}}]
    with pytest.raises(RuntimeError, match="no_tools_boundary_violation"):
        assert_no_tools_boundary(agent, {})


def test_unexpected_provider_tool_call_cannot_execute():
    import run_agent
    agent = object.__new__(run_agent.AIAgent)
    agent.tool_policy = "no_tools"
    with pytest.raises(RuntimeError, match="no_tools_boundary_violation"):
        agent._execute_tool_calls(SimpleNamespace(tool_calls=[object()]), [], "test")


def test_tampered_receipt_blocks_dispatch(tmp_path, monkeypatch):
    import json
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = _agent()
    path = start_no_tools_receipt(agent, "hello")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["tools_enabled"] = True
    path.write_text(json.dumps(data), encoding="utf-8")
    def forbidden_client(*args, **kwargs):
        pytest.fail("provider client created with tampered receipt")
    with pytest.raises(RuntimeError, match="no_tools_receipt_invalid"):
        _dispatch_nonstreaming_api_request(
            agent, {"model": "gpt-test", "messages": []},
            make_client=forbidden_client,
        )


def test_streaming_schema_injection_blocks_before_network():
    agent = _agent()
    agent._interrupt_requested = False
    with pytest.raises(RuntimeError, match="no_tools_boundary_violation"):
        interruptible_streaming_api_call(agent, {
            "tools": [{"type": "function", "function": {"name": "web_search"}}]
        })


def test_agent_initialization_skips_default_and_plugin_resolution(monkeypatch):
    import run_agent
    import hermes_cli.plugins
    def forbidden(*args, **kwargs):
        pytest.fail("default/plugin tool resolver ran in no-tools mode")
    monkeypatch.setattr(run_agent, "get_tool_definitions", forbidden)
    monkeypatch.setattr(hermes_cli.plugins, "discover_plugins", forbidden)
    agent = run_agent.AIAgent(
        model="gpt-test", provider="openai-codex",
        api_mode="codex_responses",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="test-token", no_tools=True, quiet_mode=True,
        skip_context_files=True, skip_memory=True,
    )
    assert agent.tool_policy == "no_tools"
    assert agent.enabled_toolsets == []
    assert agent.tools == []
    assert agent.valid_tool_names == set()
    assert agent._skip_mcp_refresh is True
    request = agent._build_api_kwargs([{"role": "user", "content": "hello"}])
    assert "tools" not in request
    assert "tool_choice" not in request
    assert "parallel_tool_calls" not in request
