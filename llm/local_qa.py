import ollama

def answer_with_citations(question, retrieved_chunks):
    context = ""

    for i, c in enumerate(retrieved_chunks, start=1):
        context += (
            f"[{i}] Page {c['meta']['page']} | {c['meta']['section']}\n"
            f"{c['text']}\n\n"
        )

    prompt = f"""
You are a research assistant.
Answer the question using ONLY the context below.
Cite every claim using [chunk_number].
If the answer is not present in the context, say:
"Not found in provided documents."

Question:
{question}

Context:
{context}

Answer:
"""

    response = ollama.chat(
        model="llama3.1:8b",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]
