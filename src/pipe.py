import os
import uuid
import base64
from unstructured.partition.pdf import partition_pdf
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.schema.document import Document
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import warnings
warnings.filterwarnings("ignore")

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
output_path = "./imgs"

prompt_template = """
You are an expert in maths question solving guides that contains specification, features, etc.
Answer the question based only on the following context, which can include text, images and tables:
{context}
Question: {question}
Don't answer if you are not sure and decline to answer and say "Sorry, I don't have much information about it."
Just return the helpful answer in as much as detailed possible.
Answer:
"""

prompt = ChatPromptTemplate.from_template(prompt_template)
model = ChatGoogleGenerativeAI(temperature=0, model="gemini-1.5-pro")

# Get elements
def get_elements(file_path):
    raw_pdf_elements = partition_pdf(
        filename= file_path,
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        extract_image_block_output_dir=output_path,)
    return raw_pdf_elements

def extract_elements(raw_pdf_elements):
    # Get text summaries and table summaries
    text_elements = []
    table_elements = []

    text_summaries = []
    table_summaries = []
    summary_prompt = """
    Summarize the following {element_type}:
    {element}
    """
    model = ChatGoogleGenerativeAI(temperature=0, model="gemini-1.5-pro")
    prompt = ChatPromptTemplate.from_template(summary_prompt)

    summary_chain = {"element_type": lambda x: x["element_type"], "element": lambda x: x["element"]} | prompt | model | StrOutputParser()

    for e in raw_pdf_elements:
        if 'CompositeElement' in repr(e):
            text_elements.append(e.text)
            summary = summary_chain.invoke({'element_type': 'text', 'element': e})
            text_summaries.append(summary)

        elif 'Table' in repr(e):
            table_elements.append(e.text)
            summary = summary_chain.invoke({'element_type': 'table', 'element': e})
            table_summaries.append(summary)
    return table_elements, text_elements, table_summaries, text_summaries



# Get image summaries
image_elements = []
image_summaries = []

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def summarize_image(encoded_image):
    prompt = [
        SystemMessage(content="You are a bot that is good at analyzing images."),
        HumanMessage(content=[
            {
                "type": "text",
                "text": "Describe the contents of this image."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}"
                },
            },
        ])
    ]
    for i in os.listdir(output_path):
        if i.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(output_path, i)
            encoded_image = encode_image(image_path)
            image_elements.append(encoded_image)
            summary = summarize_image(encoded_image)
            image_summaries.append(summary)
    response = ChatGoogleGenerativeAI(model="gemini-1.5-flash").invoke(prompt)
    return response.content

def image_context():
    pass



def vectorize(text_elements, text_summaries, table_elements, table_summaries):
    # Create Documents and Vectorstore
    documents = []
    retrieve_contents = []

    for e, s in zip(text_elements, text_summaries):
        i = str(uuid.uuid4())
        doc = Document(
            page_content = s,
            metadata = {
                'id': i,
                'type': 'text',
                'original_content': e
            }
        )
        retrieve_contents.append((i, e))
        documents.append(doc)

    for e, s in zip(table_elements, table_summaries):
        doc = Document(
            page_content = s,
            metadata = {
                'id': i,
                'type': 'table',
                'original_content': e
            }
        )
        retrieve_contents.append((i, e))
        documents.append(doc)

    for e, s in zip(image_elements, image_summaries):
        doc = Document(
            page_content = s,
            metadata = {
                'id': i,
                'type': 'image',
                'original_content': e
            }
        )
        retrieve_contents.append((i, s))
        documents.append(doc)

    vectorstore = FAISS.from_documents(documents=documents, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
    vectorstore.save_local("faiss_indexs")
    return vectorstore

def answer(question, vectorstore):
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
    print(result)
    return result, relevant_images
