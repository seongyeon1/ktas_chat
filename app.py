# 설치 필요: streamlit, openai 라이브러리 설치
# pip install streamlit openai

import streamlit as st
from openai import OpenAI

# OpenAI API 키 설정 (당신의 API 키를 입력하세요)
api_key = "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"

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")

# Streamlit 페이지 설정
st.title("KTAS 등급 및 응급처치 방법 알기")

# 사용자가 입력한 질문
query = st.text_input("질문을 입력하세요:", "두통환자의 KTAS 등급을 알려주세요")

# 대화 기록 설정
messages = [
    {
        "role": "system",
        "content": (
            "You are an artificial intelligence assistant and you need to "
            "engage in a helpful, detailed, polite conversation with a user."
        ),
    },
    {
        "role": "user",
        "content": query,
    },
]

# 채팅 버튼을 누르면 응답을 가져옵니다.
if st.button("답변 받기"):
    with st.spinner("AI가 응답 중입니다..."):
        # Chat Completion API 호출 (스트리밍 모드)
        response_stream = client.chat.completions.create(
            model="llama-3.1-sonar-large-128k-online",
            messages=messages,
            stream=True,
        )
        # 응답 스트리밍
        response_text = ""  # 전체 응답을 저장할 변수
        response_placeholder = st.empty()  # Streamlit의 임시 출력 공간

        for response in response_stream:
            if hasattr(response.choices[0].delta, "content"):
                response_text += response.choices[0].delta.content
                response_placeholder.text(response_text)  # 점진적으로 텍스트 출력