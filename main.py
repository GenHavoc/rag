import ollama
from typing import List, Dict

def answer_with_citations(
    question: str,
    retrieved_chunks: List[Dict],
    model: str = "llama3.1:8b"
) -> str:
    """
    Answers a question using retrieved chunks with strict citations.
    """

    # Build context
    context_blocks = []
    for i, c in enumerate(retrieved_chunks, start=1):
        context_blocks.append(
            f"[{i}] Page {c['meta']['page']} | {c['meta']['section']}\n"
            f"{c['text']}"
        )

    context = "\n\n".join(context_blocks)

    system_prompt = """
You are a careful research assistant.
Rules you MUST follow:
1. Use ONLY the provided context.
2. Every factual statement MUST have a citation like [1], [2].
3. Do NOT combine citations.
4. If the answer is missing, say exactly:
   "Not found in provided documents."
5. Do NOT use outside knowledge.
"""

    user_prompt = f"""
Question:
{question}

Context:
{context}

Answer:
"""

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()}
        ],
        options={
            "temperature": 0.1,
            "top_p": 0.9
        }
    )

    return response["message"]["content"]
