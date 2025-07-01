# Python Practice

This repository contains simple Python scripts. The `pinecone.py` example demonstrates how to use LlamaIndex with Pinecone to store and query documents.

## Environment Variables

The script relies on the following environment variables:

- `OPENAI_API_KEY` – API key for OpenAI models used by LlamaIndex.
- `PINECONE_API_KEY` – API key for Pinecone vector storage.
- `PINECONE_ENVIRONMENT` – (optional) Pinecone environment name, defaults to `us-east1-gcp`.

Set these variables before running the demo.

## Requirements

Install dependencies with `pip`:

```bash
pip install llama-index pinecone-client openai
```
