# Contributing to PRISM

PRISM is built for Forward Deployed Engineers who need production-grade
epistemic observability tooling. Contributions are welcome.

---

## Adding a New Model Adapter

The fastest way to contribute is adding a new AI model adapter.

**Step 1 — Create the adapter file:**

    adapters/your_provider/adapter.py

**Step 2 — Implement the base interface:**

Subclass `BaseAdapter` from `adapters/base/base.py`:

    from adapters.base.base import BaseAdapter

    class YourProviderAdapter(BaseAdapter):

        def __init__(self, api_key: str, model: str):
            self.client = YourProviderClient(api_key=api_key)
            self.model  = model

        def complete(self, prompt: str) -> str:
            response = self.client.complete(prompt=prompt)
            return response.text

        def get_output_distribution(self, prompt: str, hypotheses: list) -> dict:
            # Return probability distribution over hypotheses
            # Used by VERDICT for groundedness scoring
            pass

        def get_chain_of_thought(self, prompt: str) -> str:
            # Return chain-of-thought trace if available
            pass

**Step 3 — Register in `adapters/__init__.py`**

**Step 4 — Add tests in `tests/`**

**Step 5 — Submit a pull request**

---

## Running Tests

    pytest tests/ -v

All 187 tests must pass before merging.

---

## Code Style

- Python 3.11+
- Pydantic v2 for all models
- Type hints on all functions
- Docstrings on all classes and public methods

---

## Reporting Issues

Open a GitHub issue with:

- PRISM version
- Python version
- Minimal reproduction case
- Expected vs actual behavior