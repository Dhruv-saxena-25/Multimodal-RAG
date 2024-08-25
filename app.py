from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vectorstore =FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization= True)

prompt_template = """You are an expert in maths question solving that contains specification, features, etc.
Answer the question based only on the following context, which can include text, images and tables:
{context}
Question: {question}
Don't answer if you are not sure and decline to answer and say "Sorry, I don't have much information about it."
Just return the helpful answer in as much as detailed possible.
Answer:

"""
model=ChatGoogleGenerativeAI(model="gemini-1.5-flash", max_tokens=1024)
prompt=ChatPromptTemplate.from_template(prompt_template)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/get_answer")
async def get_answer(question: str = Form(...)):
    relevant_docs = vectorstore.similarity_search(question)
    context = ""
    relevant_images = []
    for d in relevant_docs:
        if d.metadata['type'] == 'text':
            context += '[text]' + d.metadata['original_content']
        elif d.metadata['type'] == 'table':
            context += '[table]' + d.metadata['original_content']
        elif d.metadata['type'] == 'image':
            context += '[image]' + d.page_content
            relevant_images.append(d.metadata['original_content'])
    chain = (
    {"context": lambda x: context, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser())
    result= chain.invoke(question)
    return JSONResponse({"relevant_images": relevant_images[0], "result": result})

