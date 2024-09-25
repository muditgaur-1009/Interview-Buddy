# import streamlit as st
# from streamlit_chat import message
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# from langchain.memory import ConversationBufferWindowMemory
# from langchain.evaluation import CriteriaEvalChain
# from langchain.agents import AgentType, initialize_agent, Tool
# from transformers import pipeline
# import sounddevice as sd
# import numpy as np
# import torch
# from gtts import gTTS
# import tempfile

# load_dotenv()
# os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # Initialize STT model on GPU if available
# device = "cuda" if torch.cuda.is_available() else "cpu"
# stt = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h", device=0 if device == "cuda" else -1)

# def text_to_speech(text):
#     tts = gTTS(text=text, lang='en', slow=False)
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
#         tts.save(fp.name)
#         return fp.name

# def speech_to_text():
#     duration = 30  # Record for 30 seconds
#     fs = 16000  # Sample rate
#     recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
#     sd.wait()
#     audio = np.squeeze(recording)
#     result = stt(audio)
#     return result["text"]

# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# def get_job_description_text(job_desc_doc):
#     text = ""
#     pdf_reader = PdfReader(job_desc_doc)
#     for page in pdf_reader.pages:
#         text += page.extract_text()
#     return text

# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     chunks = text_splitter.split_text(text)
#     return chunks

# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")

# def create_criteria_eval_chain():
#     llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
    
#     criteria = {
#         "relevance": "Is the content relevant to the job description and the question asked?",
#         "specificity": "Is the content specific and detailed enough?",
#         "clarity": "Is the content clear and easy to understand?",
#         "depth": "Does the content demonstrate depth of knowledge or experience?",
#         "conciseness": "Is the content concise without unnecessary information?",
#     }
    
#     return CriteriaEvalChain.from_llm(llm=llm, criteria=criteria)

# def evaluate_content(eval_chain, content, question, job_description):
#     result = eval_chain.evaluate_strings(
#         prediction=content,
#         input=f"Question: {question}\nJob Description: {job_description}",
#         reference="The content should be relevant, specific, clear, demonstrate depth, and be concise in relation to the question and job description."
#     )
#     return result

# def run_agentic_rag(question: str) -> str:
#     enhanced_question = f"""Using the information contained in your knowledge base, which you can access with the 'retriever' tool,
# give a comprehensive answer to the question below.
# Respond only to the question asked, response should be concise and relevant to the question.
# If you cannot find information, do not give up and try calling your retriever again with different arguments!
# Make sure to have covered the question completely by calling the retriever tool several times with semantically different queries.
# Your queries should not be questions but affirmative form sentences: e.g. rather than "how was your experience at your internship what do you learn there?", query should be "Can you describe your experience during your internship? What were your key responsibilities 
# and what specific skills or knowledge did you gain from this experience?".

# Question:
# {question}"""

#     try:
#         result = agent({"input": enhanced_question})
#         return result["output"]
#     except ValueError as e:
#         print(f"An error occurred: {e}")
#         return "I apologize, but I encountered an error while processing your request. Could you please rephrase your question or provide more context?"

# def hr_interview_bot():
#     st.title("Interactive HR Interview Bot")
    
#     if 'memory' not in st.session_state:
#         st.session_state.memory = ConversationBufferWindowMemory(k=20)
    
#     if 'evaluation_results' not in st.session_state:
#         st.session_state.evaluation_results = []
    
#     if 'chat_history' not in st.session_state:
#         st.session_state.chat_history = []
    
#     if 'eval_chain' not in st.session_state:
#         st.session_state.eval_chain = create_criteria_eval_chain()

#     with st.sidebar:
#         st.title("Upload Documents")
#         resume_docs = st.file_uploader("Upload your Resume (PDF)", accept_multiple_files=True)
#         job_desc_doc = st.file_uploader("Upload Job Description (PDF)", type="pdf")
#         job_title = st.text_input("Enter the job title you're applying for")
        
#         if st.button("Submit & Process"):
#             if resume_docs and job_desc_doc and job_title:
#                 with st.spinner("Processing..."):
#                     resume_text = get_pdf_text(resume_docs)
#                     job_desc_text = get_job_description_text(job_desc_doc)
#                     combined_text = f"Resume: {resume_text}\n\nJob Description: {job_desc_text}\n\nJob Title: {job_title}"
#                     text_chunks = get_text_chunks(combined_text)
#                     get_vector_store(text_chunks)
#                     st.session_state.vector_store_ready = True
#                     st.session_state.job_description = job_desc_text
#                     st.success("Done")
#                     st.session_state.chat_history = []
#                     st.session_state.question_count = 0
#             else:
#                 st.warning("Please upload both resume and job description, and enter the job title.")

#     if 'vector_store_ready' not in st.session_state:
#         st.session_state.vector_store_ready = False

#     if st.session_state.vector_store_ready:
#         embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#         vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

#         tools = [
#             Tool(
#                 name="retriever",
#                 func=vector_store.as_retriever().get_relevant_documents,
#                 description="Retrieve relevant documents based on the query"
#             )
#         ]

#         # Define the prompt template
#         prompt_template = PromptTemplate(
#             input_variables=["input", "agent_scratchpad"],
#             template="""You are an HR Interview Agent. Your task is to ask meaningful and relevant interview questions based on the candidate's resume and job description provided.
#             Use the tools available to retrieve relevant information and generate thoughtful questions that can evaluate the candidate's suitability for the job.
#             Ensure that your questions cover various aspects of the job role and the candidate's experience. Do not hesitate to use the retriever tool multiple times to gather all necessary information.

#             Human: {input}

#             {agent_scratchpad}

#             AI: """
#         )

#         llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)

#         global agent
#         agent = initialize_agent(
#             tools,
#             llm,
#             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#             verbose=True,
#             handle_parsing_errors=True,
#             max_iterations=3,
#             early_stopping_method="generate",
#             agent_kwargs={"prompt": prompt_template}
#         )

#         # Display chat history
#         for i, chat in enumerate(st.session_state.chat_history):
#             message(chat["content"], is_user=chat["is_user"], key=f"chat_{i}")

#         # Generate new question if needed
#         if len(st.session_state.chat_history) % 2 == 0 and st.session_state.question_count < 5:
#             new_question = run_agentic_rag("Generate an interview question based on the resume and job description.")
#             st.session_state.memory.save_context({"input": "HR"}, {"output": new_question})
            
#             # Evaluate the generated question
#             question_eval = evaluate_content(st.session_state.eval_chain, new_question, "Generate an interview question", st.session_state.job_description)
#             st.session_state.evaluation_results.append(("Question", question_eval))

#             st.session_state.chat_history.append({"content": new_question, "is_user": False})
#             st.session_state.question_count += 1

#             # Text-to-Speech for the generated question
#             tts_file = text_to_speech(new_question)
#             st.audio(tts_file, format='audio/mp3')
#             st.rerun()

#         # Handle user response
#         user_input = st.text_input("Your answer:")
#         if st.button("Submit Answer") or st.session_state.get('speech_to_text_triggered'):
#             if st.session_state.get('speech_to_text_triggered'):
#                 user_input = speech_to_text()
#                 st.text_area("Your answer:", value=user_input, key="user_input_display")

#             if user_input:
#                 st.session_state.chat_history.append({"content": user_input, "is_user": True})
#                 st.session_state.memory.save_context({"input": "User"}, {"output": user_input})

#                 # Evaluate the user response
#                 response_eval = evaluate_content(st.session_state.eval_chain, user_input, st.session_state.chat_history[-2]["content"], st.session_state.job_description)
#                 st.session_state.evaluation_results.append(("Response", response_eval))
#                 st.rerun()

# if __name__ == "__main__":
#     hr_interview_bot()


import streamlit as st
from streamlit_chat import message
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.memory import ConversationBufferWindowMemory
from langchain.evaluation import CriteriaEvalChain
from langchain.agents import AgentType, initialize_agent, Tool
from transformers import pipeline
import sounddevice as sd
import numpy as np
import torch
from gtts import gTTS
import tempfile

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize STT model on GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
stt = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h", device=0 if device == "cuda" else -1)

def text_to_speech(text):
    tts = gTTS(text=text, lang='en', slow=False)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
    return fp.name

def speech_to_text():
    duration = 30  # Record for 30 seconds
    fs = 16000  # Sample rate
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    audio = np.squeeze(recording)
    result = stt(audio)
    return result["text"]

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_job_description_text(job_desc_doc):
    text = ""
    pdf_reader = PdfReader(job_desc_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def create_criteria_eval_chain():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
    criteria = {
        "relevance": "Is the content relevant to the job description and the question asked?",
        "specificity": "Is the content specific and detailed enough?",
        "clarity": "Is the content clear and easy to understand?",
        "depth": "Does the content demonstrate depth of knowledge or experience?",
        "conciseness": "Is the content concise without unnecessary information?"
    }
    return CriteriaEvalChain.from_llm(llm=llm, criteria=criteria)

def evaluate_content(eval_chain, content, question, job_description):
    result = eval_chain.evaluate_strings(
        prediction=content,
        input=f"Question: {question}\nJob Description: {job_description}",
        reference="The content should be relevant, specific, clear, demonstrate depth, and be concise in relation to the question and job description."
    )
    return result

def run_agentic_rag(question: str) -> str:
    enhanced_question = f"""Using the information contained in your knowledge base, which you can access with the 'retriever' tool, give a comprehensive answer to the question below. Respond only to the question asked, response should be concise and relevant to the question. If you cannot find information, do not give up and try calling your retriever again with different arguments! Make sure to have covered the question completely by calling the retriever tool several times with semantically different queries. Your queries should not be questions but affirmative form sentences: e.g. rather than "how was your experience at your internship what do you learn there?", query should be "Can you describe your experience during your internship?... What were your key responsibilities and what specific skills or knowledge did you gain from this experience?". Question: {question}"""
    try:
        result = agent({"input": enhanced_question})
        return result["output"]
    except ValueError as e:
        print(f"An error occurred: {e}")
        return "I apologize, but I encountered an error while processing your request. Could you please rephrase your question or provide more context?"

def hr_interview_bot():
    st.title("Interactive HR Interview Bot")

    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(k=20)

    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = []

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'eval_chain' not in st.session_state:
        st.session_state.eval_chain = create_criteria_eval_chain()

    with st.sidebar:
        st.title("Upload Documents")
        resume_docs = st.file_uploader("Upload your Resume (PDF)", accept_multiple_files=True)
        job_desc_doc = st.file_uploader("Upload Job Description (PDF)", type="pdf")
        job_title = st.text_input("Enter the job title you're applying for")

        if st.button("Submit & Process"):
            if resume_docs and job_desc_doc and job_title:
                with st.spinner("Processing..."):
                    resume_text = get_pdf_text(resume_docs)
                    job_desc_text = get_job_description_text(job_desc_doc)
                    combined_text = f"Resume: {resume_text}\n\nJob Description: {job_desc_text}\n\nJob Title: {job_title}"
                    text_chunks = get_text_chunks(combined_text)
                    get_vector_store(text_chunks)
                    st.session_state.vector_store_ready = True
                    st.session_state.job_description = job_desc_text
                st.success("Done")
                st.session_state.chat_history = []
                st.session_state.question_count = 0
            else:
                st.warning("Please upload both resume and job description, and enter the job title.")

    if 'vector_store_ready' not in st.session_state:
        st.session_state.vector_store_ready = False

    if st.session_state.vector_store_ready:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

        tools = [
            Tool(
                name="retriever",
                func=vector_store.as_retriever().get_relevant_documents,
                description="Retrieve relevant documents based on the query"
            )
        ]

        # Define the prompt template
        prompt_template = PromptTemplate(
            input_variables=["input", "agent_scratchpad"],
            template="""You are an HR Interview Agent. Your task is to ask meaningful and relevant interview questions based on the candidate's resume and job description provided. Use the tools available to retrieve relevant information and generate thoughtful questions that can evaluate the candidate's suitability for the job. Ensure that your questions cover various aspects of the job role and the candidate's experience. Do not hesitate to use the retriever tool multiple times to gather all necessary information. Human: {input} {agent_scratchpad} AI: """
        )

        llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)

        global agent
        agent = initialize_agent(
            tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True,
            handle_parsing_errors=True, max_iterations=3, early_stopping_method="generate",
            agent_kwargs={"prompt": prompt_template}
        )

        # Display chat history
        for i, chat in enumerate(st.session_state.chat_history):
            message(chat["content"], is_user=chat["is_user"], key=f"chat_{i}")

        # Generate new question if needed
        if len(st.session_state.chat_history) % 2 == 0 and st.session_state.question_count < 5:
            new_question = run_agentic_rag("Generate an interview question based on the resume and job description.")
            st.session_state.memory.save_context({"input": "HR"}, {"output": new_question})

            # Evaluate the generated question
            question_eval = evaluate_content(st.session_state.eval_chain, new_question, "Generate an interview question", st.session_state.job_description)
            st.session_state.evaluation_results.append(("Question", question_eval))
            st.session_state.chat_history.append({"content": new_question, "is_user": False})
            st.session_state.question_count += 1

            # Text-to-Speech for the generated question
            tts_file = text_to_speech(new_question)
            st.audio(tts_file, format='audio/mp3')

        st.rerun()

        # Handle user response
        user_input = st.text_input("Your answer:")
        if st.button("Submit Answer") or st.session_state.get('speech_to_text_triggered'):
            if st.session_state.get('speech_to_text_triggered'):
                user_input = speech_to_text()
            st.text_area("Your answer:", value=user_input, key="user_input_display")
            if user_input:
                st.session_state.chat_history.append({"content": user_input, "is_user": True})
                st.session_state.memory.save_context({"input": "User"}, {"output": user_input})

                # Evaluate the user response
                response_eval = evaluate_content(st.session_state.eval_chain, user_input, st.session_state.chat_history[-2]["content"], st.session_state.job_description)
                st.session_state.evaluation_results.append(("Response", response_eval))

            st.rerun()

if __name__ == "__main__":
    hr_interview_bot()