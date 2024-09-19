import streamlit as st
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
from langchain.evaluation import load_evaluator
from langchain.evaluation.criteria import LabeledCriteria

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

def evaluate_question(question, job_description):
    evaluator = load_evaluator("labeled_criteria", criteria={
        "relevance": LabeledCriteria("Is the question relevant to the job description?"),
        "specificity": LabeledCriteria("Is the question specific enough to elicit a detailed response?"),
        "clarity": LabeledCriteria("Is the question clear and easy to understand?")
    })
    
    eval_result = evaluator.evaluate_strings(
        prediction=question,
        input=job_description,
        reference="The question should be relevant, specific, and clear in relation to the job description."
    )
    
    return eval_result

def evaluate_answer(answer, question, job_description):
    evaluator = load_evaluator("labeled_criteria", criteria={
        "relevance": LabeledCriteria("Is the answer relevant to the question and job description?"),
        "completeness": LabeledCriteria("Does the answer fully address the question?"),
        "clarity": LabeledCriteria("Is the answer clear and well-articulated?"),
        "depth": LabeledCriteria("Does the answer demonstrate depth of knowledge or experience?")
    })
    
    eval_result = evaluator.evaluate_strings(
        prediction=answer,
        input=f"Question: {question}\nJob Description: {job_description}",
        reference="The answer should be relevant, complete, clear, and demonstrate depth in relation to the question and job description."
    )
    
    return eval_result

def hr_interview_bot():
    st.title("HR Interview Bot")
    
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(k=20)
    
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = []

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
            else:
                st.warning("Please upload both resume and job description, and enter the job title.")

    if 'vector_store_ready' not in st.session_state:
        st.session_state.vector_store_ready = False

    if 'current_question' not in st.session_state:
        st.session_state.current_question = None

    if 'question_count' not in st.session_state:
        st.session_state.question_count = 0

    if st.session_state.vector_store_ready:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local("faiss_index", embeddings)

        if st.session_state.current_question is None:
            st.session_state.current_question = generate_question(vector_store, st.session_state.memory)
            st.session_state.memory.save_context({"input": "HR"}, {"output": st.session_state.current_question})
            
            # Evaluate the generated question
            question_eval = evaluate_question(st.session_state.current_question, st.session_state.job_description)
            st.session_state.evaluation_results.append(("Question", question_eval))

        st.write(f"Question {st.session_state.question_count + 1}: {st.session_state.current_question}")
        user_answer = st.text_area(f"Your answer to Question {st.session_state.question_count + 1}", key=f"q{st.session_state.question_count + 1}")

        if st.button("Next Question"):
            if user_answer:
                st.session_state.memory.save_context({"input": "Candidate"}, {"output": user_answer})
                
                # Evaluate the user's answer
                answer_eval = evaluate_answer(user_answer, st.session_state.current_question, st.session_state.job_description)
                st.session_state.evaluation_results.append(("Answer", answer_eval))
                
                st.session_state.question_count += 1
                if st.session_state.question_count < 5:
                    st.session_state.current_question = generate_question(vector_store, st.session_state.memory)
                    st.session_state.memory.save_context({"input": "HR"}, {"output": st.session_state.current_question})
                    
                    # Evaluate the new question
                    question_eval = evaluate_question(st.session_state.current_question, st.session_state.job_description)
                    st.session_state.evaluation_results.append(("Question", question_eval))
                else:
                    st.session_state.current_question = None
                st.experimental_rerun()
            else:
                st.warning("Please provide an answer before moving to the next question.")

        if st.session_state.question_count >= 5:
            if st.button("Generate Interview Summary and Evaluation"):
                summary_prompt = f"""
                Based on the following interview conversation and evaluation results, provide a comprehensive report on the candidate's performance:

                Conversation history:
                {st.session_state.memory.buffer}

                Evaluation results:
                {st.session_state.evaluation_results}

                Please provide:
                1. A summary of the candidate's background, strengths, and potential fit for the role.
                2. An assessment of their proficiency in answering questions, including relevance, completeness, clarity, and depth of responses.
                3. An evaluation of the quality of the interview questions, including relevance, specificity, and clarity.
                4. Highlights of both positive aspects and areas for improvement.
                5. Overall recommendation based on the interview performance and evaluation metrics.
                """

                model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
                summary = model.predict(summary_prompt)
                
                st.write("Interview Summary and Evaluation:")
                st.write(summary)

def main():
    st.set_page_config("HR Interview Bot with Evaluation Metrics")
    st.header("HR Interview Bot powered by Gemini💁 with Evaluation Metrics")
    
    hr_interview_bot()

if __name__ == "__main__":
    main()