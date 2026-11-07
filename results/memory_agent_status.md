# Memory Agent & Long-Context QA Status Report

## Status

**Memory agent: not implemented, evaluation planned (long-context QA)**

## Codebase Audit Findings

- **Repository Inspection**: Audited all source files in the repository (`model_core.py`, `app.py`, `prompts.py`, `tests/test_veriagent.py`).
- **Implementation State**: There is no Memory Agent class, no conversation buffer manager, no vector index / embeddings store, and no retrieval mechanism in the current codebase.
- **Pipeline Integration**: The reasoning routes in `model_core.py` (Numeric Math, Mathematical Concept, General/Identity) operate as stateless request-response pipelines.
- **Test Coverage**: No unit tests, integration tests, or benchmarks exist for multi-turn conversational memory or long-context document QA.

## Exactly What Remains to Implement

1. **Conversational Memory Module**: Implementation of a sliding-window or summarization-based context manager for multi-turn dialog state.
2. **Vector Retrieval Augmentation (RAG)**: Integration of a local embedding model (e.g. `nomic-embed-text` via Ollama) and vector index (e.g. SQLite-VSS or ChromaDB) for document-grounded question answering.
3. **Long-Context Evaluation Suite**: Benchmarking on a standard long-context evaluation dataset (e.g., L-Eval or NarrativeQA) using deterministic needle-in-a-haystack verification.
