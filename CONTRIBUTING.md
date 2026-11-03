# Contributing

## Adding a framework

1. `src/agentparity/adapters/<framework>.py`: subclass `Adapter`. Read the `Adapter` docstring in `core/adapter.py`; it is the spec.
   - Map `context_id` to the framework's session or thread.
   - Call `ctx.authorize_tool(name, args)` from the framework's **native** pre-tool hook, before the tool body runs. On a deny, the body must not run.
   - Claim only what you implement. Unchecked claims fail the suite.
2. `src/agentparity/conformance/fixtures/<framework>.py`: a fixture that builds the seven behaviours in `conformance/spec.py`. Use a scripted model, not a real one. TOOL and ASK must go through the framework's real tool machinery. A fixture must never call `authorize_tool` itself.
3. `tests/test_<framework>.py`: `pytest.importorskip(...)`, then assert `run_suite(YourFixture()).ok`.
4. Add an extra to `pyproject.toml`, a row to the README matrix and a CI job.

## Rules for the suite

- Never weaken a check to get a PASS. A real framework limitation is a valid result: set the capability to `False` and document why.
- Every new check needs a broken variant in `tests/test_conformance_suite.py` that it fails. A check you have not watched fail proves nothing.

## Development

Frameworks pin conflicting dependencies (for example `openai-agents` and `crewai` need different `openai` majors), so give each one its own virtualenv. CI does the same.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[a2a,dev,langgraph]"
pytest
```

By contributing you agree that your contributions are licensed under Apache-2.0.
