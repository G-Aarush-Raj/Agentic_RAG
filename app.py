import json
import math
from agentapps import Agent, Tool
from agentapps.model import GrokChat

STORE_PATH = "data/vector_store/store.json"

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b + 1e-10)

class RAGSearchTool(Tool):
    def __init__(self):
        super().__init__(
            name="rag_search",
            description="Search pre-ingested documents using vector similarity"
        )
        with open(STORE_PATH, "r", encoding="utf-8") as f:
            self.store = json.load(f)

    def execute(self, query_embedding: list, top_k: int = 5) -> str:
        scored = []

        for item in self.store:
            score = cosine_similarity(query_embedding, item["embedding"])
            scored.append((score, item["text"]))

        scored.sort(key=lambda x: x[0], reverse=True)
        return "\n\n".join(text for _, text in scored[:top_k])

    def get_parameters(self):
        return {
            "type": "object",
            "properties": {
                "query_embedding": {
                    "type": "array",
                    "items": {"type": "number"}
                },
                "top_k": {"type": "integer"}
            },
            "required": ["query_embedding"]
        }

grok_model = GrokChat(
    id="grok-3-mini",
    api_key="GROK_API_KEY"
)


rag_agent = Agent(
    name="RAG Agent",
    role="Answer questions using retrieved document context",
    model=grok_model,
    tools=[RAGSearchTool()],
    instructions=[
        "When the user asks a question:",
        "1. Convert the question into a semantic embedding (array of numbers)",
        "2. Call the rag_search tool with that embedding",
        "3. Use ONLY the returned context to answer",
        "4. If the answer is not present, say 'I don't know'",
        "Never hallucinate"
    ],
    show_tool_calls=True
)


if __name__ == "__main__":
    print("📚 AgentApps RAG Agent (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ("exit", "quit"):
            break

        response = rag_agent.print_response(user_input)
        print(response)
