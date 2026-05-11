from langchain_community.document_loaders import PyPDFLoader

def load_pdf(path):
    loader = PyPDFLoader(path)
    docs = loader.load()

    pages = []
    for d in docs:
        pages.append({
            "page": d.metadata.get("page", 0) + 1,
            "text": d.page_content
        })

    return pages
