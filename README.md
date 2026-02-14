## Agentic_RAG
AgenticRAG is a lightweight, agent-first Retrieval-Augmented Generation system built with AgentApps and custom tools. It cleanly separates document ingestion from runtime retrieval, performs tool-driven search over precomputed embeddings, and ensures answers are generated strictly from retrieved context without hallucination.

## ✨ Key Features
- 🧠 Agent-driven RAG using AgentApps
- 🔧 Custom retrieval tool (tool-based search)
- 📦 Lightweight agent runtime (no FAISS, no torch, no transformers)
- 🔁 Deterministic document ingestion
- 🚫 No hallucination (answers strictly from retrieved context)
- 🧱 Clean, production-friendly architecture

## ⚙️ Requirements
Python **3.9+**
Install dependencies:
pip install -r requirements.txt
