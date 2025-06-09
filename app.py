import streamlit as st
from streamlit_chat import message
from vectorstore import get_vectorstore_from_pdf
from chatbot import get_pdf_conversational_chain, is_similar_to_previous_question

st.set_page_config(page_title="📄 PDF 기반 GPT 챗봇 (RAG)", layout="centered")
st.title("PDF 기반 GPT 챗봇 (LangChain RAG)")

# 업로드 UI
uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type="pdf")

# PDF 업로드 처리
if uploaded_file:
    with open(f"data/uploaded_docs/{uploaded_file.name}", "wb") as f:
        f.write(uploaded_file.read())
    st.success("PDF 업로드 완료!")

    # vectorstore와 chain을 세션에 저장
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = get_vectorstore_from_pdf(f"data/uploaded_docs/{uploaded_file.name}")
        st.session_state.chain = get_pdf_conversational_chain(st.session_state.vectorstore)
        st.session_state.chat_history = []

# 사용자 질문
user_input = st.text_input("문서에 대해 질문해보세요!", key="input")

# 질문 처리
if user_input:
    if "vectorstore" not in st.session_state:
        st.error("먼저 PDF 파일을 업로드해주세요.")
    else:
        # 유사 질문 여부 확인
        is_similar, similar_question = is_similar_to_previous_question(
            user_input,
            st.session_state.chat_history,
            st.session_state.vectorstore,
            threshold=0.9 # 너무 다 유사하다고 나와서 수정

        )

        if is_similar:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.chat_history.append({
                "role": "bot",
                "content": f"이 질문은 이전에 유사한 질문이 있었습니다:\n\"{similar_question}\""
            })
        else:
            result = st.session_state.chain.run(user_input)
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.chat_history.append({"role": "bot", "content": result})

# 메시지 출력
if "chat_history" in st.session_state:
    for i, msg in enumerate(st.session_state.chat_history):
        is_user = msg["role"] == "user"
        message(msg["content"], is_user=is_user, key=f"msg_{i}")

else:
    st.info("📂 먼저 PDF 파일을 업로드해주세요.")
