"""Exact-query policy through the real AIAgent turn and request builder."""

import hashlib
import io
import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _agent(tmp_path, monkeypatch, *, api_mode="codex_responses", audited=True, audited_text="exact"):
    import run_agent

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    provider = "openai-codex" if api_mode == "codex_responses" else "custom"
    base_url = (
        "https://chatgpt.com/backend-api/codex"
        if api_mode == "codex_responses" else "https://example.invalid/v1"
    )
    agent = run_agent.AIAgent(
        model="gpt-test", provider=provider,
        api_mode=api_mode, base_url=base_url,
        api_key="test-token", no_tools=True, audited_query_transport=audited,
        audited_query_text=audited_text if audited else None,
        quiet_mode=True, skip_context_files=True, skip_memory=True,
        skip_background_review=True, platform="cli", max_iterations=2,
    )
    return agent


def test_real_codex_turn_preserves_exact_query_and_skips_enrichment(tmp_path, monkeypatch):
    import cli
    import hermes_cli.lifecycle as lifecycle
    import hermes_cli.plugins as plugins

    query = " \n日本語 exact query\n "
    agent = _agent(tmp_path, monkeypatch, audited_text=query)
    memory = MagicMock()
    memory.prefetch_all.return_value = "POST_RACE_MEMORY_SENTINEL"
    memory.build_system_prompt.return_value = "POST_RACE_MEMORY_SENTINEL"
    agent._memory_manager = memory
    plugin_prompt = MagicMock(return_value=[{"content": "PLUGIN_CONTEXT_SENTINEL"}])
    monkeypatch.setattr(plugins, "render_system_prompt_sections", plugin_prompt)
    hook_calls = []

    def hook(name, **kwargs):
        if name == "pre_llm_call":
            hook_calls.append(name)
            return [{"context": "PLUGIN_CONTEXT_SENTINEL"}]
        return []

    monkeypatch.setattr(lifecycle, "invoke_hook", hook)
    requests = []
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **kw: SimpleNamespace(close=lambda: None))

    def provider(request, **kwargs):
        requests.append(request)
        return SimpleNamespace(output=[], output_text="answer", status="completed", usage=None)

    monkeypatch.setattr(agent, "_run_codex_stream", provider)
    result = cli._run_cli_agent_turn(agent, user_message=query, conversation_history=[])
    assert result["final_response"] == "answer"
    assert len(requests) == 1
    assert requests[0]["input"] == [{"role": "user", "content": query}]
    assert "POST_RACE_MEMORY_SENTINEL" not in requests[0]["instructions"]
    assert "PLUGIN_CONTEXT_SENTINEL" not in requests[0]["instructions"]
    assert not requests[0].get("tools")
    assert memory.prefetch_all.call_count == 0
    assert memory.on_turn_start.call_count == 0
    assert memory.build_system_prompt.call_count == 0
    plugin_prompt.assert_not_called()
    assert hook_calls == []
    receipts = list((tmp_path / "receipts").glob("no-tools-*.json"))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text(encoding="utf-8"))["external_send_state"] == "dispatch_started"


def test_gateway_note_fails_closed_before_provider(tmp_path, monkeypatch):
    import cli

    agent = _agent(tmp_path, monkeypatch, audited_text="query")
    agent._gateway_turn_context_notes = "GATEWAY_NOTE_SENTINEL"
    network = MagicMock(side_effect=AssertionError("provider called"))
    monkeypatch.setattr(agent, "_run_codex_stream", network)
    with pytest.raises(RuntimeError, match="audited_query_gateway_notes_forbidden"):
        cli._run_cli_agent_turn(agent, user_message="query", conversation_history=[])
    network.assert_not_called()


def test_final_codex_content_guard_blocks_tamper(tmp_path, monkeypatch):
    import cli

    agent = _agent(tmp_path, monkeypatch)
    network = MagicMock(side_effect=AssertionError("provider called"))
    monkeypatch.setattr(agent, "_run_codex_stream", network)
    original = agent._build_api_kwargs

    def tampered(messages, *args, **kwargs):
        request = original(messages, *args, **kwargs)
        request["input"][-1]["content"] = "POST_VALIDATION_TAMPER"
        return request

    monkeypatch.setattr(agent, "_build_api_kwargs", tampered)
    with pytest.raises(RuntimeError, match="audited_query_content_mismatch"):
        cli._run_cli_agent_turn(agent, user_message="exact", conversation_history=[])
    network.assert_not_called()


def test_chat_completion_final_content_guard_blocks_tamper(tmp_path, monkeypatch):
    from agent.no_tools_boundary import assert_audited_query_content

    agent = _agent(tmp_path, monkeypatch, api_mode="chat_completions")
    assert_audited_query_content(agent, {"messages": [{"role": "user", "content": "exact"}]})
    with pytest.raises(RuntimeError, match="audited_query_content_mismatch"):
        assert_audited_query_content(agent, {"messages": [{"role": "user", "content": "tampered"}]})


def test_real_chat_completions_preserves_query(tmp_path, monkeypatch):
    import cli

    query = " \nchat exact\n "
    agent = _agent(tmp_path, monkeypatch, api_mode="chat_completions", audited_text=query)
    requests = []

    def create(**request):
        requests.append(request)
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="answer", tool_calls=None),
                finish_reason="stop",
            )], usage=None,
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=lambda: None)
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **kw: client)
    result = cli._run_cli_agent_turn(agent, user_message=query, conversation_history=[])
    assert result["final_response"] == "answer"
    assert len(requests) == 1
    assert requests[0]["messages"][-1]["content"] == query
    assert not requests[0].get("tools")


def test_real_agent_from_query_stdin_quiet_cli(tmp_path, monkeypatch, capsys):
    import cli
    import hermes_cli.main as main_mod
    from hermes_cli._parser import build_top_level_parser

    query = " \nexact WU6 query\n "
    raw = query.encode("utf-8")
    sha = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8"))
    monkeypatch.setattr(main_mod, "_has_any_provider_configured", lambda: True)
    monkeypatch.setattr(main_mod, "_pin_kanban_board_env", lambda: None)
    monkeypatch.setattr(main_mod, "_sync_bundled_skills_for_startup", lambda: None)
    monkeypatch.setattr(main_mod, "_confirm_startup_expensive_model_override", lambda args: None)
    monkeypatch.setattr(cli, "_finalize_single_query", lambda shell: None)
    requests = []

    class Shell:
        def __init__(self, **kwargs):
            assert kwargs["audited_query_transport"] is True
            self.agent = _agent(tmp_path, monkeypatch, audited_text=kwargs["audited_query_text"])
            self.session_id = self.agent.session_id
            self.conversation_history = []
            self._active_agent_route_signature = None
            monkeypatch.setattr(
                self.agent, "_create_request_openai_client",
                lambda **kw: SimpleNamespace(close=lambda: None),
            )
            monkeypatch.setattr(
                self.agent, "_run_codex_stream",
                lambda request, **kw: requests.append(request) or SimpleNamespace(
                    output=[], output_text="answer", status="completed", usage=None
                ),
            )

        def _claim_active_session(self, *args, **kwargs):
            return True

        def _ensure_runtime_credentials(self):
            return True

        def _resolve_turn_agent_config(self, query):
            return {"signature": None, "model": None, "runtime": None}

        def _init_agent(self, **kwargs):
            return True

    monkeypatch.setattr(cli, "HermesCLI", Shell)
    args = build_top_level_parser()[0].parse_args([
        "chat", "--query-stdin", "--expected-query-sha256", sha,
        "-Q", "--provider", "openai-codex", "-m", "gpt-test",
        "--no-tools", "--safe-mode", "--source", "tool", "--max-turns", "20",
    ])
    with pytest.raises(SystemExit) as exc:
        main_mod.cmd_chat(args)
    assert exc.value.code == 0
    assert len(requests) == 1
    assert requests[0]["input"] == [{"role": "user", "content": query}]
    assert not requests[0].get("tools")
    output = capsys.readouterr()
    assert output.err.count("HERMES_QUERY_STDIN_RECEIPT ") == 1
    assert output.out.strip() == "answer"


def test_ordinary_no_tools_keeps_user_context_behavior(tmp_path, monkeypatch):
    import cli
    import hermes_cli.lifecycle as lifecycle

    agent = _agent(tmp_path, monkeypatch, audited=False)
    memory = MagicMock()
    memory.prefetch_all.return_value = "POST_RACE_MEMORY_SENTINEL"
    memory.build_system_prompt.return_value = ""
    agent._memory_manager = memory
    monkeypatch.setattr(
        lifecycle, "invoke_hook",
        lambda name, **kwargs: ["PLUGIN_CONTEXT_SENTINEL"] if name == "pre_llm_call" else [],
    )
    requests = []
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **kw: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(
        agent, "_run_codex_stream",
        lambda request, **kw: requests.append(request) or SimpleNamespace(
            output=[], output_text="answer", status="completed", usage=None
        ),
    )
    result = cli._run_cli_agent_turn(agent, user_message="ordinary question", conversation_history=[])
    assert result["final_response"] == "answer"
    assert len(requests) == 1
    content = requests[0]["input"][0]["content"]
    assert "POST_RACE_MEMORY_SENTINEL" in content
    assert "PLUGIN_CONTEXT_SENTINEL" in content
    assert not requests[0].get("tools")


def test_codex_wire_boundary_rejects_late_relay_mutation(tmp_path, monkeypatch):
    from agent import codex_runtime, relay_llm

    agent = _agent(tmp_path, monkeypatch)
    request = agent._build_api_kwargs([{"role": "user", "content": "exact"}])
    physical_send = MagicMock(side_effect=AssertionError("network called"))
    client = SimpleNamespace(responses=SimpleNamespace(create=physical_send))

    def relay(request, opener, **kwargs):
        mutated = dict(request)
        mutated["input"] = [{"role": "user", "content": "LATE_RELAY_TAMPER"}]
        return opener(mutated)

    monkeypatch.setattr(relay_llm, "stream", relay)
    with pytest.raises(RuntimeError, match="audited_query_content_mismatch"):
        codex_runtime.run_codex_stream(agent, request, client=client)
    physical_send.assert_not_called()


def test_cli_rejects_query_changed_after_stdin_validation():
    import cli

    with pytest.raises(RuntimeError, match="audited_query_content_mismatch"):
        cli.main(
            query="POST_VALIDATION_TAMPER", query_stdin=True,
            audited_query_transport=True, audited_query_text="validated stdin",
            no_tools=True, quiet=True,
        )
