"""No-tools CLI startup must not discover plugins or connect MCP servers."""

import pytest


def test_cold_cli_import_suppresses_legacy_plugin_discovery(tmp_path, monkeypatch):
    import os
    import subprocess
    import sys

    code = """
from hermes_cli import plugins
calls = []
def forbidden(*args, **kwargs):
    calls.append(True)
    raise AssertionError('plugin discovery during no-tools CLI import')
plugins.discover_plugins = forbidden
with plugins.suppress_plugin_discovery_for_no_tools_import():
    import cli
assert calls == []
assert plugins._skip_discovery_for_no_tools_import.get() is False
"""
    env = os.environ.copy()
    env["HERMES_HOME"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, "-B", "-c", code], env=env,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("deferred", [False, True])
def test_real_cli_agent_setup_skips_discovery(tmp_path, monkeypatch, deferred):
    import cli
    import run_agent
    import hermes_cli.plugins as plugins
    import hermes_cli.mcp_startup as mcp_startup
    from hermes_cli.main import _prepare_agent_startup
    from hermes_cli._parser import build_top_level_parser

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    if deferred:
        monkeypatch.setenv("HERMES_DEFER_AGENT_STARTUP", "1")
    else:
        monkeypatch.delenv("HERMES_DEFER_AGENT_STARTUP", raising=False)
    monkeypatch.setattr(cli, "_deferred_agent_startup_done", False)
    calls = {"plugin": 0, "background_mcp": 0, "pre_agent_mcp": 0}

    def forbidden(kind):
        def reject(*args, **kwargs):
            calls[kind] += 1
            raise AssertionError(f"{kind} called")
        return reject

    monkeypatch.setattr(plugins, "discover_plugins", forbidden("plugin"))
    monkeypatch.setattr(mcp_startup, "start_background_mcp_discovery", forbidden("background_mcp"))
    monkeypatch.setattr(mcp_startup, "ensure_mcp_discovery_before_agent_build", forbidden("pre_agent_mcp"))
    args = build_top_level_parser()[0].parse_args(["chat", "-Q", "-q", "hello", "--no-tools"])
    _prepare_agent_startup(args)

    shell = cli.HermesCLI(
        model="gpt-test", provider="openai-codex", api_key="test-token",
        base_url="https://chatgpt.com/backend-api/codex",
        no_tools=True, audited_query_transport=False, compact=True,
    )
    monkeypatch.setattr(shell, "_ensure_runtime_credentials", lambda: True)
    assert shell._init_agent()
    assert isinstance(shell.agent, run_agent.AIAgent)
    assert shell.agent.tool_policy == "no_tools"
    assert calls == {"plugin": 0, "background_mcp": 0, "pre_agent_mcp": 0}


def test_deferred_normal_startup_still_discovers(monkeypatch):
    import cli
    import hermes_cli.plugins as plugins
    import hermes_cli.mcp_startup as mcp_startup

    monkeypatch.setenv("HERMES_DEFER_AGENT_STARTUP", "1")
    monkeypatch.setattr(cli, "_deferred_agent_startup_done", False)
    calls = {"plugin": 0, "mcp": 0}
    monkeypatch.setattr(plugins, "discover_plugins", lambda: calls.__setitem__("plugin", calls["plugin"] + 1))
    monkeypatch.setattr(mcp_startup, "start_background_mcp_discovery", lambda **kw: calls.__setitem__("mcp", calls["mcp"] + 1))
    cli._prepare_deferred_agent_startup(tool_policy="no_tools")
    assert calls == {"plugin": 0, "mcp": 0}
    cli._prepare_deferred_agent_startup(tool_policy="auto")
    assert calls == {"plugin": 1, "mcp": 1}
