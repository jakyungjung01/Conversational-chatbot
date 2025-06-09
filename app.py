'''
Stremlit ui화면
'''

import streamlit as st
from streamlit_chat import message
from chabot import get_qa_chain
from vectorstore import get_vectorstore_from_pdf

st.set_page_config(page_title="챗봇시스템", layout="centered")
st.title("langchain chatbot")

qa_chain = get_qa_chain()

uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type="pdf")

if uploaded_file:
    with open(f"data/uploaded_docs/{uploaded_file.name}", "wb") as f:
        f.write(uploaded_file.read())
    st.success(" PDF 업로드 완료!")

    # 체인 초기화
    if "chain" not in st.session_state:
        vectorstore = get_vectorstore_from_pdf(f"data/uploaded_docs/{uploaded_file.name}")
        st.session_state.chain = get_qa_chain()
        st.session_state.chat_history = []



user_input = st.text_input("질문을 입력하세요", key="input")
if user_input:
    st.session_state.messages.append({"role":"user", "content": user_input})
    response = qa_chain.invoke(user_input)
    
    st.session_state.messages.append({"role": "bot", "content": response})
    
for i, msg in enumerate(st.session_state.messages):
    is_user = msg["role"] == "user"
    message(msg["content"], is_user=is_user, key=f"msg_{i}")
    