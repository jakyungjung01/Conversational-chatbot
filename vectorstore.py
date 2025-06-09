from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

'''
def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    # 간단한 텍스트 데이터 로딩
    with open("data/sample_docs.txt", "r") as f:
        docs = [Document(page_content=line.strip()) for line in f if line.strip()]
    
    # FAISS 인덱스 생성
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore
'''

def get_vectorstore_from_pdf(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    #텍스트 분할
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(pages)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore