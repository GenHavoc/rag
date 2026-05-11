from langchain.text_splitter import TokenTextSplitter

def chunk_text(text, chunk_size=400, overlap=50):
    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        encoding_name="cl100k_base"
    )
    return splitter.split_text(text)
