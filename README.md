## 1. 프로젝트 개요

본 프로젝트는 **금융 용어**에 대한 사용자의 질문에 정확하고 신뢰성 있는 답변을 제공하는 **검색 증강 생성(RAG)** 기반 챗봇의 기술 검증(Proof of Concept) 프로젝트입니다. Hugging Face의 금융 Q&A 데이터셋(`aiqwe/FinShibainu`)을 기반으로 **VectorDB를 구축**하고, Langchain 프레임워크와 OpenAI의 GPT-4o 모델을 활용하여 **RAG 파이프라인**을 설계 및 구현했습니다.


## 2. 핵심 기능 및 특징

*   **금융 데이터 처리 및 VectorDB 구축 (`담당할 업무: vectorDB 구축`):**
    *   금융 Q&A 데이터셋(`aiqwe/FinShibainu`) 로드 및 Pandas를 이용한 전처리 수행.
    *   **한국어 및 금융 도메인 특화 임베딩 모델**(BGE-M3 기반 `seongil-dn/...-finance-50`)을 사용하여 데이터 벡터화 및 **ChromaDB** 기반 VectorDB 구축.
    *   Google Drive 연동을 통해 구축된 VectorDB의 영속성 확보.
    *   데이터 길이 분포 분석(Matplotlib) 및 **한국어 특화 토크나이저**(`EbanLee/kobart-summary-v3`)를 활용한 토큰화 및 길이 제한(512 토큰) 처리.
*   **Langchain 기반 RAG 파이프라인 설계:**
    *   VectorDB 내 금융 문서 대상 유사도 기반 검색(Similarity Search) Retriever 구성.
    *   **상세 프롬프트 엔지니어링** :
        *   `PromptTemplate`을 활용하여 챗봇의 역할(금융 전문가), 페르소나(친근한 말투), 제약 조건(모를 시 인정), 응답 강화(예시 활용, 추가 질문 유도) 등 상세 가이드라인을 포함한 프롬프트 작성.
    *   OpenAI GPT-4o 모델과 Retriever를 결합한 `RetrievalQA` 체인 구성.
*   **검색 품질 관리 및 개선 시도 :**
    *   유사도 점수 기반 결과 필터링 로직 구현 (임계값 설정).
    *   (실험적) 금융 용어 유사어 사전을 통한 쿼리 확장 기능 구현 시도 (Recall 향상 목적).
*   **인터랙티브 데모 :**
    *   `Gradio`를 활용하여 사용자가 직접 질문하고 답변을 확인할 수 있는 웹 기반 데모 인터페이스 구축.

## 3. 기술 스택

*   **언어:** `Python 3.11`
*   **핵심 프레임워크/라이브러리:**
    *   `Langchain`, `Langchain-OpenAI`, `Langchain-Community`
    *   `OpenAI` (GPT-4o)
    *   `Hugging Face Transformers`, `Datasets`, `Tokenizers`
    *   `ChromaDB`, `FAISS` (설치됨)
    *   `Pandas`, `Matplotlib`, `Tiktoken`
    *   `Gradio` (v3.50.2)
*   **임베딩 모델:**
    *   BGE-M3 기반 한국어/금융 특화 모델 (`seongil-dn/bge-m3-kor-retrieval...`)
    *   OpenAI Embeddings (실험적 사용)
*   **실행 환경:** Google Colab (GPU)
*   **데이터:** Hugging Face (`aiqwe/FinShibainu`, 시사경제용어사전 필터링)

## 4. 보유기술 및 핵심역량


*   ** RAG 챗봇 서비스 기획 (프롬프트 작성, vectorDB 구축):**
    *   **VectorDB 구축:** 금융 데이터 특성을 고려한 임베딩 모델 선정부터 ChromaDB 구축, 저장까지 전 과정 수행.
    *   **프롬프트 작성:** 챗봇의 역할, 어조, 행동 지침을 상세히 정의하는 프롬프트 엔지니어링 경험 보유.
*   ** LLM 이용 운영 업무 자동화:** 상세 프롬프트 작성 능력은 LLM 기반 자동화 태스크 설계에 직접 적용 가능.
*   ** 신규 상담 콘텐츠/학습 데이터 작성:** 외부 데이터셋(Hugging Face)을 로드하고 RAG 시스템에 적합하도록 분석, 전처리, 필터링하는 역량 시연.
*   ** 고객 발화 분석 및 학습 개선 사항 도출:** 유사어 확장, 유사도 스코어 필터링 등 검색/응답 품질 개선을 위한 아이디어 구현 및 실험 경험.
*   ** Python, 데이터 분석 툴 이용:** 프로젝트 전반에 `Python`을 능숙하게 사용하며 `Pandas`, `Matplotlib` 등 데이터 처리 및 분석 라이브러리 활용 능력 보유.
*   ** 자연어 처리 기본 개념 이해:** 임베딩, 토크나이저, VectorDB, RAG 등 핵심 NLP 개념을 이해하고 실제 코드에 적용 (`우대사항`). 특히, 한국어 및 금융 도메인에 적합한 모델(Tokenizer, Embedding)을 선정하여 적용.

## 5. 주요 코드 하이라이트

*   **도메인 특화 임베딩 모델 활용:** 일반 모델 대신 한국어와 금융 분야 검색에 특화된 **BGE-M3 기반 모델**(`seongil-dn/bge-m3-kor-retrieval...`)을 선택하여 VectorDB 구축, RAG 성능 최적화를 시도했습니다.
*   **상세 지침 기반 프롬프트 엔지니어링:** 단순 Q&A를 넘어 챗봇의 페르소나, 정보 부족 시 대응, 예시 활용, 추가 질문 유도 등 구체적인 행동 지침을 포함한 **체계적인 프롬프트**를 `PromptTemplate`으로 설계했습니다.
*   **토큰 길이 분석 및 처리:** 데이터 시각화(Matplotlib)를 통해 텍스트 길이를 분석하고, **한국어 토크나이저**를 사용하여 최대 입력 길이에 맞춰 데이터를 안전하게 처리하는 로직을 구현했습니다.

## 6. 실행 방법 (Gradio 데모)

1.  제공된 Jupyter Notebook (`vectorDB (1).ipynb`)을 Google Colab 환경에서 엽니다.
2.  **OpenAI API 키 설정:** Notebook 내 `os.environ["OPENAI_API_KEY"] = ...` 부분을 본인의 키로 설정합니다.
3.  Notebook의 모든 셀을 순차적으로 실행합니다.
4.  마지막 셀(`import gradio as gr`)을 실행하면 데모 UI가 실행되고 **Public URL**이 출력됩니다.
5.  출력된 URL에 접속하여 금융 용어 관련 질문을 입력하고 챗봇의 답변을 확인합니다.
