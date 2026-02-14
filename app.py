from flask import Flask, request,render_template
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import pdfplumber
from markupsafe import Markup
load_dotenv()


#single_resume -> making your resume better
#bulk_resume -> shortlist resumes for a job description
app = Flask(__name__)

groq = ChatGroq(api_key=os.getenv('GROQ_API_KEY'),model="llama-3.1-8b-instant")
def response_cleaner(response):
    response = Markup(response.replace("\n", "<br>"))
    response = Markup(response.replace("**", " "))
    return response
@app.route("/")
def home():
    name = 'Maheshwar varma'
    return render_template("index.html", name=name)

@app.route("/upload")
def upload():
    return render_template("upload_resume.html")

@app.route("/upload_resume", methods=["POST"])
def upload_resume():
    
    job_description = request.files.get('job_description')
    resumes = request.files.get("resume")
    extracted_description = []
    extracted_texts = []
    with pdfplumber.open(job_description) as pdf:
        extracted_description = "\n".join(page.extract_text() for page in pdf.pages)

    print("Job Description Extracted:", extracted_description)
    print("----"*50)

    with pdfplumber.open(resumes) as pdf:
        text = "\n".join(page.extract_text() for page in pdf.pages)
        extracted_texts.append(text)
    print("Resumes Extracted:", extracted_texts)

    prompt_template = """You are an expert HR professional. Given the job description and resume, 
    identify and suggest the changes that best match the job description. 

    Job Description:
    {extracted_description}

    Resume:
    {extracted_texts}
    Provide a detailed analysis of the resume in relation to the job description, highlighting areas of improvement and suggesting specific changes to enhance the resume's alignment with the job requirements.
    """

    prompt = PromptTemplate(
        input_variables=["extracted_description", "extracted_texts"],
        template=prompt_template
    )

    model = LLMChain(
        llm=groq,
        prompt=prompt
    )

    resumes_combined = "\n\n".join(extracted_texts)
    response = model.run(extracted_description=extracted_description, extracted_texts=resumes_combined)
    return render_template("result.html", response=response_cleaner(response))
if __name__ == "__main__":
    app.run(debug=True,port=3000)