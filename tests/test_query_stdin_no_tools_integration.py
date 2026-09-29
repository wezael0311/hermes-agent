"""Audited stdin input and quiet no-tools CLI integration, with no network."""

import hashlib
import io
import json
import sys
from types import SimpleNamespace

import pytest

from hermes_cli._parser import build_top_level_parser


def _parse(extra):
    return build_top_level_parser()[0].parse_args(["chat", *extra])


def _run(tmp_path, monkeypatch, capsys, raw, extra, *, injection=None):
    import cli as cli_mod
    import hermes_cli.main as main_mod
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_IGNORE_USER_CONFIG", "1")
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8"))
    monkeypatch.setattr(main_mod, "_has_any_provider_configured", lambda: True)
    monkeypatch.setattr(main_mod, "_pin_kanban_board_env", lambda: None)
    monkeypatch.setattr(main_mod, "_sync_bundled_skills_for_startup", lambda: None)
    monkeypatch.setattr(main_mod, "_confirm_startup_expensive_model_override", lambda args: None)
    monkeypatch.setattr(cli_mod, "_finalize_single_query", lambda shell: None)

    observed = {"provider_calls": 0, "queries": [], "payloads": []}

    class FakeAgent:
        tool_policy = "no_tools"
        enabled_toolsets = []
        tools = []
        api_mode = "chat_completions"
        provider = "openai-codex"
        model = "gpt-test"

        def run_conversation(self, *, user_message, **kwargs):
            observed["queries"].append(user_message)
            receipts = list((tmp_path / "receipts").glob("no-tools-*.json"))
            assert len(receipts) == 1
            before = json.loads(receipts[0].read_text(encoding="utf-8"))
            assert before["external_send_state"] == "not_sent"
            assert before["effective_toolsets"] == []
            assert before["effective_tool_count"] == 0
            request = {"model": self.model, "messages": [{"role": "user", "content": user_message}]}
            request.update(injection or {})

            def make_client(*args, **kwargs):
                observed["provider_calls"] += 1
                client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
                    create=lambda **payload: observed["payloads"].append(payload) or "ok"
                )))
                return client

            _dispatch_nonstreaming_api_request(self, request, make_client=make_client)
            return {"final_response": "ok"}

    class FakeShell:
        def __init__(self, **kwargs):
            observed["shell_kwargs"] = kwargs
            self.agent = FakeAgent()
            self.session_id = "test-session"
            self.conversation_history = []
            self._active_agent_route_signature = None

        def _claim_active_session(self, *args, **kwargs):
            return True

        def _ensure_runtime_credentials(self):
            return True

        def _resolve_turn_agent_config(self, query):
            return {"signature": None, "model": None, "runtime": None}

        def _init_agent(self, **kwargs):
            return True

    monkeypatch.setattr(cli_mod, "HermesCLI", FakeShell)
    args = _parse(extra)
    with pytest.raises(SystemExit) as exc:
        main_mod.cmd_chat(args)
    return exc.value.code, observed, capsys.readouterr()


def _command(sha, *extra):
    return [
        "--query-stdin", "--expected-query-sha256", sha, "-Q",
        "--provider", "openai-codex", "-m", "gpt-test", "--no-tools",
        "--safe-mode", "--source", "tool", "--max-turns", "20", *extra,
    ]


def test_combined_quiet_transport_reaches_mock_provider(tmp_path, monkeypatch, capsys):
    raw = '日本語 "quote" $(whoami) `backtick`\n'.encode("utf-8")
    sha = hashlib.sha256(raw).hexdigest()
    code, seen, output = _run(tmp_path, monkeypatch, capsys, raw, _command(sha))
    assert code == 0
    assert seen["queries"] == [raw.decode("utf-8")]
    assert seen["provider_calls"] == 1
    assert len(seen["payloads"]) == 1
    assert not any(k in seen["payloads"][0] for k in ("tools", "tool_choice", "parallel_tool_calls"))
    lines = [line for line in output.err.splitlines() if line.startswith("HERMES_QUERY_STDIN_RECEIPT ")]
    assert len(lines) == 1
    transport = json.loads(lines[0].split(" ", 1)[1])
    assert transport == {
        "transport_version": "hermes-chat-query-stdin-v1",
        "query_byte_count": len(raw), "query_sha256": sha,
        "utf8_decode_success": True, "expected_sha_match": True,
        "external_send_performed": False,
    }
    receipts = list((tmp_path / "receipts").glob("no-tools-*.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert receipt["external_send_state"] == "dispatch_started"
    assert receipt["query_sha256"] == sha


def test_quiet_no_tools_without_stdin_starts_one_receipt(tmp_path, monkeypatch, capsys):
    code, seen, output = _run(
        tmp_path, monkeypatch, capsys, b"", ["-q", "hello", "-Q", "--no-tools"]
    )
    assert code == 0
    assert seen["queries"] == ["hello"]
    assert seen["provider_calls"] == 1
    receipts = list((tmp_path / "receipts").glob("no-tools-*.json"))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text(encoding="utf-8"))["external_send_state"] == "dispatch_started"
    assert "HERMES_QUERY_STDIN_RECEIPT " not in output.err


def test_stdin_query_is_not_rewritten_as_image_attachment(tmp_path, monkeypatch, capsys):
    raw = b"C:\\pictures\\example.png"
    sha = hashlib.sha256(raw).hexdigest()
    code, seen, _ = _run(tmp_path, monkeypatch, capsys, raw, _command(sha))
    assert code == 0
    assert seen["queries"] == [raw.decode("utf-8")]


@pytest.mark.parametrize("raw,sha,error", [
    (b"wrong", hashlib.sha256(b"right").hexdigest(), "query_stdin_sha256_mismatch"),
    (b"\xff", hashlib.sha256(b"\xff").hexdigest(), "query_stdin_invalid_utf8"),
    (b"text", "ABC", "query_stdin_expected_sha256_invalid"),
])
def test_stdin_rejected_before_agent(tmp_path, monkeypatch, capsys, raw, sha, error):
    code, seen, output = _run(tmp_path, monkeypatch, capsys, raw, _command(sha))
    assert code == 2
    assert error in output.err
    assert seen["provider_calls"] == 0
    assert not list((tmp_path / "receipts").glob("*.json"))


def test_injected_tool_schema_blocks_before_client(tmp_path, monkeypatch, capsys):
    raw = b"hello"
    sha = hashlib.sha256(raw).hexdigest()
    with pytest.raises(RuntimeError, match="no_tools_boundary_violation"):
        _run(tmp_path, monkeypatch, capsys, raw, _command(sha), injection={"tools": [{"type": "function", "function": {"name": "bad"}}]})
    assert list((tmp_path / "receipts").glob("no-tools-*.json"))


@pytest.mark.parametrize("args", [
    ["-q", "other"], ["--query-file", "other.txt"], ["--image", "a.png"], ["--tui"],
])
def test_query_stdin_conflicts_at_parser_or_preflight(tmp_path, monkeypatch, capsys, args):
    raw = b"hello"
    sha = hashlib.sha256(raw).hexdigest()
    if args[0] in ("-q", "--query-file"):
        with pytest.raises(SystemExit) as exc:
            _parse(_command(sha, *args))
        assert exc.value.code == 2
    else:
        code, seen, _ = _run(tmp_path, monkeypatch, capsys, raw, _command(sha, *args))
        assert code == 2
        assert seen["provider_calls"] == 0


def test_sha_option_requires_query_stdin():
    from hermes_cli.main import cmd_chat
    with pytest.raises(SystemExit) as exc:
        cmd_chat(_parse(["--expected-query-sha256", "0" * 64, "-Q"]))
    assert exc.value.code == 2


def test_normal_toolset_parser_regression():
    from hermes_cli.main import _prepare_agent_startup
    assert _parse(["-q", "hello", "-t", "web"]).toolsets == "web"
    assert _parse(["-q", "hello"]).toolsets is None
    with pytest.raises(SystemExit):
        _prepare_agent_startup(_parse(["-q", "hello", "--no-tools", "-t", "web"]))
