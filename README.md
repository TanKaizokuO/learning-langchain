# LangChain Examples

A small collection of example scripts and demos showing how to use LangChain and related components.

## Quickstart

Prerequisites: Python 3.8+ and git.

Create and activate a virtual environment, then install requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running examples

Run any example script directly, for example:

```bash
python basic-rag/ingestion.py
python basic-search-agent/main.py
python langchain-models/test.py
```

Note: Some scripts may require API keys or additional setup; check the top of each file for details.

## Repository layout

- [basic-rag](basic-rag) — simple retrieval-augmented generation examples.
- [basic-search-agent](basic-search-agent) — search agent examples.
- [E-commerce_Agent](E-commerce_Agent) — e-commerce demo agents.
- [langchain-chains](langchain-chains) — chain examples (sequential, parallel, conditional).
- [langchain-document-loaders](langchain-document-loaders) — loaders for text, CSV, PDF, web, etc.
- [langchain-models](langchain-models) — model and embedding examples.
- [langchain-prompts](langchain-prompts) — prompt templates and utilities.
- [langchain-runnables](langchain-runnables) — runnable composition examples.
- [langchain-text-splitters](langchain-text-splitters) — text splitting strategies.
- [langchain-structured-output](langchain-structured-output) — structured output and schema examples.
- [langchain-output-parsers](langchain-output-parsers) — output parser examples.

## Notes

- Use the provided `.venv` pattern above to keep dependencies isolated.
- If you run into missing-dependency errors, install packages listed in the relevant subfolder `requirements.txt` files.
