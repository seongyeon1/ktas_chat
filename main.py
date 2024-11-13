import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from typing import Sequence
from typing_extensions import Annotated, TypedDict

# 환경 변수 로드
load_dotenv()

import base64

# Streamlit 앱 제목
st.title("응급처치 질문 응답 시스템")

# 로고 이미지 표시 함수
def show_logo():
    logo_path = "logo.png"
    try:
        # 이미지 로드 및 Base64 인코딩
        with open(logo_path, "rb") as img_file:
            logo_base64 = base64.b64encode(img_file.read()).decode("utf-8")
        
        # CSS를 통해 로고를 오른쪽 상단에 고정
        st.markdown(
            f"""
            <style>
            .fixed-logo {{
                position: fixed;
                top: 10px;
                right: 10px;
                width: 100px;
                z-index: 100;
            }}
            </style>
            <img src="data:image/png;base64,{logo_base64}" class="fixed-logo">
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.error("로고 이미지 파일을 찾을 수 없습니다. 경로를 확인하세요.")

# 로고 이미지 표시 함수 호출
show_logo()

# 모델 및 데이터베이스 설정
gemini_1_5_flash = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)

# Embeddings 모델 설정
embeddings_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

db = FAISS.load_local('./db/combined-first-aid', embeddings_model, allow_dangerous_deserialization=True)

# Retriever 설정
retriever = db.as_retriever(
    search_type='mmr',
    search_kwargs={'k': 5, 'lambda_mult': 0.15}
)

# 프롬프트 설정 (응급 상황에 대한 한글 답변)
system_prompt = (
    "당신은 응급도 분류 및 응급처치 안내를 제공하는 한국어 전용 AI 어시스턴트입니다. "
    "대화 기록과 문서 정보를 참고해서 응급도를 파악하고 응급처치 방법을 안내해 주세요. "
    "사용자 질문에 대해 모르는 내용이 있거나 확실하지 않으면 솔직하게 '해당 정보를 제공할 수 없습니다'라고 답해주세요. "
    "119에 연락이 필요한 경우 이를 명확히 알리세요."
    "\n\n대화 기록:\n{chat_history}\n\n문서 정보:\n{context}\n\n질문:\n{input}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# 응답 체인 생성
question_answer_chain = create_stuff_documents_chain(gemini_1_5_flash, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 상태 정의
class State(TypedDict):
    input: str
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    answer: str

# 대화 상태 그래프 설정
def call_model(state: State):
    # 대화 기록을 텍스트 형식으로 변환하여 프롬프트에 전달
    chat_history_text = "\n".join(
        [f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"Assistant: {msg.content}" for msg in state["chat_history"]]
    )
    
    # 문서 `context`를 retriever로 검색하여 추가
    retrieved_docs = retriever.get_relevant_documents(state["input"])
    doc_context = "\n".join([doc.page_content for doc in retrieved_docs])

    # `input`, `chat_history`, `context`를 사용해 응답 생성
    response = rag_chain.invoke({
        "input": state["input"],
        "chat_history": chat_history_text,
        "context": doc_context
    })
    
    return {
        "chat_history": state["chat_history"] + [
            HumanMessage(content=state["input"]),
            AIMessage(content=response.get("answer", "죄송합니다, 이 질문에 대한 답변을 생성할 수 없습니다. 전문가의 도움을 받으세요."))
        ],
        "context": doc_context,
        "answer": response.get("answer", "죄송합니다, 이 질문에 대한 답변을 생성할 수 없습니다. 전문가의 도움을 받으세요.")
    }

# 워크플로우 설정
workflow = StateGraph(state_schema=State)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Checkpointer 설정
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 대화 내용 출력 (챗봇 스타일)
for message in st.session_state.chat_history:
    sender = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(sender):
        st.write(message.content)

# 질문 입력창과 제출 버튼을 아래에 배치
with st.container():
    user_input = st.text_input("질문을 입력하세요", "")

    if st.button("질문하기"):
        if user_input:
            # State 객체 초기화 및 모델 호출
            state = State(
                input=user_input,
                chat_history=st.session_state.chat_history,
                context="",
                answer=""
            )
            
            # Checkpointer에 필요한 'thread_id'를 추가하여 config 설정
            config = {"configurable": {"thread_id": "abc123"}}
            result = app.invoke(state, config=config)

            # 대화 기록 업데이트
            st.session_state.chat_history = result["chat_history"]