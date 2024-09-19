from flask import Flask, render_template, request, redirect, url_for, session
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from langchain_google_genai import ChatGoogleGenerativeAI

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


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

app = Flask(__name__)
app.secret_key = 'your_secret_key'

BUFFER_SIZE = 20

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
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

def generate_question(vector_store, previous_questions):
    question_prompt = f"""
Given the following context, which includes the candidate's resume, the job description, and the job title,
generate a thoughtful HR interview question that pertains specifically to the candidate's qualifications and the job requirements.
Ensure the question is open-ended, allowing the candidate to elaborate on their skills, experiences, and qualifications. 
Avoid repeating any of these previously asked questions: {previous_questions}.
The question should effectively gauge the candidate's experience level and alignment with the role's requirements,
considering both their background and the specific job they're applying for.
Question:
"""
    
    docs = vector_store.similarity_search(question_prompt, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.4)
    question = model.predict(context + "\n" + question_prompt)
    
    return question.strip()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        resume_docs = request.files.getlist('resume_docs')
        job_desc_doc = request.files.get('job_desc_doc')
        job_title = request.form.get('job_title')
        
        if resume_docs and job_desc_doc and job_title:
            resume_text = get_pdf_text(resume_docs)
            job_desc_text = get_pdf_text([job_desc_doc])
            combined_text = f"Resume: {resume_text}\n\nJob Description: {job_desc_text}\n\nJob Title: {job_title}"
            text_chunks = get_text_chunks(combined_text)
            get_vector_store(text_chunks)
            session['vector_store_ready'] = True
            session['previous_questions'] = []
            session['answers'] = []
            session['question_count'] = 0
            session['current_question'] = None
            session['conversation_buffer'] = []
            return redirect(url_for('interview'))
    
    return render_template('index.html')

@app.route('/interview', methods=['GET', 'POST'])
def interview():
    if 'vector_store_ready' not in session:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        user_answer = request.form.get('user_answer')
        if user_answer:
            # Add the question and answer to the buffer
            if len(session['conversation_buffer']) >= BUFFER_SIZE:
                session['conversation_buffer'].pop(0)
            session['conversation_buffer'].append({
                'question': session['current_question'],
                'answer': user_answer
            })
            
            session['answers'].append(user_answer)
            session['question_count'] += 1
            
            if session['question_count'] < 5:
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                vector_store = FAISS.load_local("faiss_index", embeddings)
                session['current_question'] = generate_question(vector_store, [q['question'] for q in session['conversation_buffer']])
                session['previous_questions'].append(session['current_question'])
            else:
                session['current_question'] = None
                return redirect(url_for('summary'))
    
    if session['current_question'] is None:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local("faiss_index", embeddings)
        session['current_question'] = generate_question(vector_store, [q['question'] for q in session['conversation_buffer']])
        session['previous_questions'].append(session['current_question'])
    
    return render_template('interview.html', question=session['current_question'], question_count=session['question_count'])

@app.route('/summary')
def summary():
    if 'vector_store_ready' not in session:
        return redirect(url_for('index'))
    
    summary_prompt = f"""
Based on the following interview answers, provide a brief report on the candidate's performance, including areas where they excelled, 
areas for improvement, and any inaccuracies or weaknesses in their responses. Consider the job description and the specific role they're applying for.
{chr(10).join([f"Question: {q['question']}{chr(10)}Answer: {q['answer']}{chr(10)}" for q in session['conversation_buffer']])}
Provide a concise paragraph summarizing the candidate's background, strengths, and potential fit for the role. Assess their proficiency in answering questions,
 the accuracy of their responses, their communication skills, clarity of thought, and problem-solving abilities. 
 Highlight both positive aspects and constructive feedback to help the candidate improve their performance.
"""
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    summary = model.predict(summary_prompt)
    
    return render_template('summary.html', summary=summary)

if __name__ == "__main__":
    app.run(debug=True)
