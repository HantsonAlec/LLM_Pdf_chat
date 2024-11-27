from pathlib import Path

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

DATA_PATH = Path(__file__).parent.parent / "data"
PDF_FILE = DATA_PATH / "1706.03762.pdf"
INDEX_PATH = DATA_PATH / "faiss_index_attention"

EMBEDDING_MODEL = OllamaEmbeddings(model="llama3")
RETRIEVAL_QA_PROMPT = hub.pull("langchain-ai/retrieval-qa-chat")
LLM_MODEL = ChatOllama(model="llama3")

if __name__ == "__main__":
    loader = PyPDFLoader(str(PDF_FILE))
    documents = loader.load()

    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )
    docs = text_splitter.split_documents(documents=documents)

    if INDEX_PATH.exists():
        vectorstore = FAISS.load_local(
            str(INDEX_PATH),
            embeddings=EMBEDDING_MODEL,
            allow_dangerous_deserialization=True,
        )
    else:
        vectorstore = FAISS.from_documents(documents=docs, embedding=EMBEDDING_MODEL)
        vectorstore.save_local(str(INDEX_PATH))

    combine_docs_chain = create_stuff_documents_chain(LLM_MODEL, RETRIEVAL_QA_PROMPT)
    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )

    query = "Explain how attention works to a 5 year old"
    result = retrieval_chain.invoke(input={"input": query})
    print(result)
