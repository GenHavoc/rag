import streamlit as st
import tempfile

from ingestion.pdf_loader import load_pdf
from ingestion.section_parser import detect_section
from ingestion.chunker import chunk_text
from embeddings.embedder import Embedder
from vectorstore.faiss_store import FaissStore
from llm.local_qa import answer_with_citations

st.set_page_config(page_title="RAG Research Assistant", layout="wide")
st.title("📄 Research Paper Q&A (Local RAG)")

uploaded_file = st.file_uploader("Upload a research paper (PDF)", type=["pdf"])

@st.cache_resource
def build_store(chunks):
    texts = [c["text"] for c in chunks]
    metadata = [{"page": c["page"], "section": c["section"]} for c in chunks]

    embedder = Embedder()
    embeddings = embedder.embed(texts)

    store = FaissStore(dim=embeddings.shape[1])
    store.add(embeddings, texts, metadata)

    return store, embedder

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    pages = load_pdf(pdf_path)

    chunks = []
    current_section = "Body"

    for p in pages:
        detected = detect_section(p["text"])
        if detected != "Body":
            current_section = detected

        for c in chunk_text(p["text"]):
            chunks.append({
                "text": c,
                "page": p["page"],
                "section": current_section
            })

    st.success(f"Loaded {len(pages)} pages → {len(chunks)} chunks")

    store, embedder = build_store(chunks)

    question = st.text_input("Ask a question about the paper:")

    if question:
        q_emb = embedder.embed([question])[0]
        results = [r for r in store.search(q_emb, k=5) if r["score"] > 0.4]

        st.subheader("🔍 Retrieved Evidence")
        for r in results:
            st.markdown(
                f"**Page {r['meta']['page']} | {r['meta']['section']}**  "
                f"(score={r['score']:.2f})\n\n{r['text'][:300]}..."
            )

        st.subheader("🧠 Answer (with citations)")
        st.markdown(answer_with_citations(question, results))
