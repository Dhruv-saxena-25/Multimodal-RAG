import os
import uuid
import base64
import tqdm as notebook_tqdm
from IPython import display
from unstructured.partition.pdf import partition_pdf
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.schema.document import Document
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
import warnings
warnings.filterwarnings("ignore")

output_path = "./imgs"

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
    response = ChatGoogleGenerativeAI(model="gemini-1.5-flash").invoke(prompt)
    return response.content

for i in os.listdir(output_path):
    if i.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(output_path, i)
        encoded_image = encode_image(image_path)
        image_elements.append(encoded_image)
        summary = summarize_image(encoded_image)
        image_summaries.append(summary)