# RETRIEVAL-AUGMENTED GENERATION (RAG) 

# streamlit run course_and_university_recommender.py
# Chat Q&A Framework for RAG Apps

# Imports 
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Import CHAT OLLAMA
from langchain_community.chat_models import ChatOllama

import streamlit as st
import yaml
import uuid

# Initialize Streamlit app
st.set_page_config(page_title="University and Course Recommender", layout="wide")
st.title("University and Course Recommender")

# Load the API Key securely
OPENAI_API_KEY = yaml.safe_load(open('credentials.yml'))['openai']

# Set up memory
import streamlit as st

# Initialize session state key
if "langchain_messages" not in st.session_state:
    st.session_state.langchain_messages = []

# Now you can use StreamlitChatMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

msgs = StreamlitChatMessageHistory(key="langchain_messages")

if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

view_messages = st.expander("View the message contents in session state")

def create_rag_chain(api_key):
    
    embedding_function = OpenAIEmbeddings(
        model='text-embedding-ada-002',
        api_key=api_key
        #chunk_size=500,
    )
    vectorstore = Chroma(
        persist_directory="data/chroma_store",
        embedding_function=embedding_function
    )
    
    retriever = vectorstore.as_retriever()
    
    llm = ChatOpenAI(
         model="gpt-3.5-turbo", 
         temperature=0.4, 
         api_key=api_key,
         max_tokens=3500,
     )

    #llm = ChatOllama(
    #    model="llama3"
    #    #model="llama3:70b"
    #   )

    # COMBINE CHAT HISTORY WITH RAG RETREIVER
    # * 1. Contextualize question: Integrates RAG
    contextualize_q_system_prompt = """
    You are a highly skilled contextualizer for a course and university recommendation system designed for Nigerian students. Your role is to rephrase or interpret user queries to align them with the dataset structure and system requirements. Ensure the reformulated query remains accurate, relevant, and adheres to the following guidelines:
    
    1. **Dataset Schema Awareness:**
   - Match queries to the columns and fields in the dataset, such as:
     - `WAEC grades` (e.g., subjects and grades: A1, B2, etc.).
     - `required_subjects` and `optional_subjects`.
     - User preferences like `location`, `region`, `institution_type`, `tuition_fee`, and `jamb_cutoff`.
   - Avoid introducing concepts, variables, or preferences that are not explicitly part of the dataset.
   
   2. **Eligibility Logic:**
   - Ensure the rephrased query focuses on students with at least 5 credit passes (A1 - C6) in relevant subjects.
   - Include logic for substituting optional subjects if required subjects do not meet the 5-credit minimum.
   
   3. **User Preferences Integration:**
   - Contextualize preferences such as:
     - Specific locations or states (e.g., Lagos, Abuja).
     - Institution types (Federal, State, Private).
     - Maximum tuition fees.
     
     4. **Clarifications and Missing Inputs:**
   - If a query lacks essential details (e.g., WAEC grades or location preferences), rephrase it to include a polite request for the missing information.
   
   5. **Tone and Style:**
   - Be concise, formal, and user-friendly.
   - Use neutral language to avoid confusion or misinterpretation.
   
   Examples:
   - Original Query: "Which course can I study with these grades? A1 in Biology, B2 in Physics, C4 in Chemistry, and C5 in English?"
   Reformulated Query: "Based on your WAEC grades (A1 in Biology, B2 in Physics, C4 in Chemistry, and C5 in English), what courses are available, ensuring required subjects are met and optional subjects are considered if needed?"
   
   - Original Query: "Can you recommend a university in Lagos with low tuition fees?"
  Reformulated Query: "What are the available universities in Lagos with tuition fees within the specified budget range?"
  
  Always ensure the rephrased query is compatible with the dataset and supports accurate recommendations.
    """

    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # * 2. Answer question based on Chat Context
    qa_system_prompt = """
    You are a course and university recommendation expert specializing in helping Nigerian students navigate their higher education options. Your primary goal is to recommend suitable courses and universities based on students' WAEC results and other preferences. \
    You are a recommender expert for university and course selection tools. \
    Use the following pieces of retrieved context to answer the question. \
    Any query that isn't related to university and program selection tool should not be given response to./
    They should be redirected to tailor their questions to fit into your given purpose. \
    If you don't know the answer, just say you don't know. \
    Your purpose is to recommend courses and universities for secondary school leaving students. \
    Answer questions that are related to courses and universities. \
    
    
    You must strictly adhere to the rules and requirements outlined below to provide accurate and reliable recommendations. \
     
    ### Key Guidelines:
    1. **Eligibility Criteria:**
    - Students must have at least 5 credit passes (A1 - C6) in their WAEC results to be eligible for any course recommendation.
    - For every course:
        - **Required Subjects:** These are compulsory subjects that must be passed for a student to be eligible for the course.
        - **Optional Subjects:** These are additional subjects that can be used to meet the minimum of 5 credit passes if the required subjects are insufficient.

    2. **Course Recommendation Logic:**
        - Ensure all required subjects for a course are passed with a credit grade (A1 - C6).
        - If the minimum of 5 credit passes is not met using the required subjects, include optional subjects to complete the credit requirement.
        - If a student fails critical required subjects (e.g., Physics for Engineering or Biology for Medicine), clearly explain why the course cannot be recommended.

    3. **University Recommendation Logic:**
        - Recommend universities based on the user's preferences:
            - **Location:** Suggest universities in the specified state (e.g., Lagos, Abuja, Imo).
            - **Institution Type:** Filter universities based on type (Federal, State, Private) as per the user's choice.
            - **Tuition Fees:** Recommend universities within the user's specified budget range.

    4. **Additional Information:**
        - Include the JAMB cutoff mark for each course and university in your recommendations.
        - Provide the official website link of each recommended university for more details.

    5. **User Interaction:**
        - Prompt users to input:
        - WAEC subjects and grades.
        - Preferred location.
        - Institution type (Federal, State, Private).
        - Maximum tuition fee they can afford.
   - Verify the completeness and validity of user inputs before generating recommendations.
   - If no suitable recommendations can be made, explain the reasons clearly and offer actionable advice (e.g., retaking certain WAEC subjects).

    6. **Tone and Style:**
        - Be friendly, professional, and supportive.
        - Use clear and simple language to explain recommendations and requirements.
        - Encourage students by highlighting available opportunities based on their qualifications.

    You must ensure that all recommendations align with the dataset provided and the criteria mentioned above. Always prioritize accuracy, relevance, and clarity in your responses. \
    

    If you don't know any answer, don't try to make up an answer. Just say that you don't know and to visit the university website for additional information.\
    
    Don't be overconfident and don't hallucinate. 
    {context}"""
            #Minimum of 5 recommendations that best fit the users input should be made initailly and more can be given as per users request. /
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # * Combine both RAG + Chat Message History
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: msgs,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

rag_chain = create_rag_chain(OPENAI_API_KEY)


# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

if question := st.chat_input("Hello :), how can I be of assistance to you?"):
    with st.spinner("Hold on please ......."):
        st.chat_message("human").write(question)     
           
        response = rag_chain.invoke(
            {"input": question}, 
            config={
                "configurable": {"session_id": "any"}
            },
        )
        # Debug response
        # print(response)
        # print("\n")
  
        st.chat_message("ai").write(response['answer'])

# * NEW: View the messages for debugging
# Draw the messages at the end, so newly generated ones show up immediately
with view_messages:
    """
    Message History initialized with:
    ```python
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    ```

    Contents of `st.session_state.langchain_messages`:
    """
    view_messages.json(st.session_state.langchain_messages)
