import streamlit as st
from streamlit_chat import message
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.memory import ConversationBufferWindowMemory
from langchain.evaluation import CriteriaEvalChain

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

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

def generate_question(vector_store, memory):
    question_prompt = f"""
Given the following context, which includes the candidate's resume, the job description, and the job title,
generate a thoughtful HR interview question that pertains specifically to the candidate's qualifications and the job requirements.
Ensure the question is open-ended, allowing the candidate to elaborate on their skills, experiences, and qualifications. 
Avoid repeating any previously asked questions.

The question should effectively gauge the candidate's experience level and alignment with the role's requirements,
considering both their background and the specific job they're applying for.

Previous conversation:
{memory.buffer}

Question:
"""
    
    docs = vector_store.similarity_search(question_prompt, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.4)
    question = model.predict(context + "\n" + question_prompt)
    
    return question.strip()

def create_criteria_eval_chain():
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
    
    criteria = {
        "relevance": "Is the content relevant to the job description and the question asked?",
        "specificity": "Is the content specific and detailed enough?",
        "clarity": "Is the content clear and easy to understand?",
        "depth": "Does the content demonstrate depth of knowledge or experience?",
        "conciseness": "Is the content concise without unnecessary information?",
    }
    
    return CriteriaEvalChain.from_llm(llm=llm, criteria=criteria)

def evaluate_content(eval_chain, content, question, job_description):
    result = eval_chain.evaluate_strings(
        prediction=content,
        input=f"Question: {question}\nJob Description: {job_description}",
        reference="The content should be relevant, specific, clear, demonstrate depth, and be concise in relation to the question and job description."
    )
    return result

def hr_interview_bot():
    st.title("HR Interview Bot")
    
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
        vector_store = FAISS.load_local("faiss_index", embeddings)

        # Display chat history
        for i, chat in enumerate(st.session_state.chat_history):
            message(chat["content"], is_user=chat["is_user"], key=f"chat_{i}")

        # Generate new question if needed
        if len(st.session_state.chat_history) % 2 == 0 and st.session_state.question_count < 5:
            new_question = generate_question(vector_store, st.session_state.memory)
            st.session_state.memory.save_context({"input": "HR"}, {"output": new_question})
            
            # Evaluate the generated question
            question_eval = evaluate_content(st.session_state.eval_chain, new_question, "Generate an interview question", st.session_state.job_description)
            st.session_state.evaluation_results.append(("Question", question_eval))

            st.session_state.chat_history.append({"content": new_question, "is_user": False})
            st.session_state.question_count += 1
            st.rerun()

        # Get user input
        user_input = st.text_input("Your answer:", key="user_input")

        if user_input:
            st.session_state.chat_history.append({"content": user_input, "is_user": True})
            st.session_state.memory.save_context({"input": "Candidate"}, {"output": user_input})
            
            # Evaluate the user's answer
            answer_eval = evaluate_content(st.session_state.eval_chain, user_input, st.session_state.chat_history[-2]["content"], st.session_state.job_description)
            st.session_state.evaluation_results.append(("Answer", answer_eval))

            st.rerun()

        # Generate summary after 5 questions
        if st.session_state.question_count >= 5 and len(st.session_state.chat_history) % 2 == 0:
            if st.button("Generate Interview Summary and Evaluation"):
                summary_prompt = f"""
                Based on the following interview conversation and evaluation results, provide a comprehensive report on the candidate's performance:

                Conversation history:
                {st.session_state.memory.buffer}

                Evaluation results:
                {st.session_state.evaluation_results}

                Please provide:
                1. A summary of the candidate's background, strengths, and potential fit for the role.
                2. An assessment of their proficiency in answering questions, including relevance, specificity, clarity, depth, and conciseness of responses.
                3. An evaluation of the quality of the interview questions, including relevance, specificity, clarity, depth, and conciseness.
                4. Highlights of both positive aspects and areas for improvement.
                5. Overall recommendation based on the interview performance and evaluation metrics.
                """

                model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
                summary = model.invoke(summary_prompt)
                
                st.write("Interview Summary and Evaluation:")
                st.write(summary)
def main():
    st.set_page_config("HR Interview Bot with Chat UI")
    st.header("HR Interview Bot powered by Gemini💁 with Chat UI")
    
    hr_interview_bot()

if __name__ == "__main__":
    main()