from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
from IPython.display import Markdown as md
import os 
from dotenv import load_dotenv
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


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
model=ChatGoogleGenerativeAI(model="gemini-1.5-flash")
prompt=ChatPromptTemplate.from_template(prompt_template)


application=Flask(__name__)
CORS(application)
app=application


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route("/get_answer",  methods=["POST"])
def get_answer():
    if request.method == "POST":
        question = request.form['question']
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
        | StrOutputParser()
        )
        if len(relevant_images) !=0:
            pass
        else:
            relevant_images = ["No Image To Display..."]
        answer= chain.invoke(question)
        return render_template("index.html", results= answer, image = relevant_images[0])
        
if __name__ == '__main__':
    app.run(debug= True, host='0.0.0.0')

