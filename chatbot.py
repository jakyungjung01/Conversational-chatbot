'''
langchain QA 체인 정의 (로컬 모델 버전)
-> 사용자의 이전 대화 히스토리를 기억하며 문맥 기반 질문 응답을 가능하게 합니다.
-> 새로운 질문이 들어올 때마다 기존 질문 이력에서 유사도를 계산하고, 유사한 질문이 있으면 해당 답변을 제공하는 로직을 추가


'''

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


def get_pdf_conversational_chain(vectorstore):
    # 로컬 모델 로드 (FLAN-T5-Small)
    model_name = "google/flan-t5-small"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.3,
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    # 대화 히스토리 저장을 위한 메모리
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # ConversationalRetrievalChain 생성
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        verbose=True,
    )

    return chain



def is_similar_to_previous_question(user_input, chat_history, vectorstore, threshold=0.9):
    # 입력 질문 임베딩
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    question_vector = embeddings.embed_query(user_input)

    # 유사한 문서와 유사도 점수 함께 검색
    similar_docs_with_score = vectorstore.similarity_search_with_score_by_vector(
        question_vector, k=1
    )

    if not similar_docs_with_score:
        return False, None

    # 가장 유사한 질문 및 유사도
    most_similar_doc, score = similar_docs_with_score[0]

    # FAISS는 거리 기반이라 score가 낮을수록 더 유사함 → 거리 → 유사도 변환 필요
    # 유사도 = 1 - 거리 (예시 적용, cosine일 경우)
    similarity = 1 - score
    
    print(f"[DEBUG] 유사도: {similarity:.4f}, 질문 내용: {most_similar_doc.page_content}")


    if similarity >= threshold:
        return True, most_similar_doc.page_content
    else:
        return False, None
