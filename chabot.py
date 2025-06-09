'''
langchain QA 체인 정의 (로컬 모델 버전)
-> 사용자의 이전 대화 히스토리를 기억하며 문맥 기반 질문 응답을 가능하게 합니다.


'''
#from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from langchain_community.llms import HuggingFacePipeline
from vectorstore import get_vectorstore_from_pdf

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

def get_qa_chain():
    # 로컬 모델 로드 (FLAN-T5-Small)
    model_name = "google/flan-t5-small"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name) #텍스트 -> 숫자로 변경

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.5,
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    # 벡터스토어 로드
    vectorstore = get_vectorstore_from_pdf()

    '''
    # QA 체인 생성
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type='stuff'
    )
    '''
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        verbose=True,
    )

    

    return qa_chain
