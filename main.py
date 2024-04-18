from fastapi import FastAPI, File, UploadFile
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import TFIDFRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_extraction_chain
from langchain.prompts.prompt import PromptTemplate
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_core.documents import Document
from mimetypes import guess_type
from pydantic import BaseModel
from openai import OpenAI
from typing import List
import logging
import uvicorn
import os

app = FastAPI()
model = ChatOpenAI(model="gpt-4", temperature=0)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

logger = logging.getLogger(__name__)

class DocumentRequest(BaseModel):
    text: str


@app.get("/")
async def read_root():
    return {"Running Successfuly!"}
    
@app.post("/documents")
async def get_documents(docx: DocumentRequest):
    
    docs = docx.text

    #### Textsplitter takes itrable therefore we need to split the text into two parts ####
    total_docs = len(docs)
    midpoint = total_docs // 2

    # Split the list into two parts
    docs_part1 = docs[:midpoint]
    docs_part2 = docs[midpoint:]

    # Create a list to store the key-value pairs
    docs_split = [
                     {"pageContent": docs_part1}
                 ] + [
                     {"pageContent": docs_part2}
                 ]

    docs_ = [Document(page_content=d["pageContent"]) for d in docs_split]

    # splitting document into chunks
    documents = RecursiveCharacterTextSplitter(
        chunk_size=3000, chunk_overlap=300
    ).split_documents(docs_) #because now we are gettings docs so need to extract the text from it

    # defining the vector store, creating embeddings
    vector = FAISS.from_documents(documents, OpenAIEmbeddings())
    retriever = vector.as_retriever(search_type="mmr")

    # defining the prompt template string
    prompt_string = """You are a very knowledgeable clinical pathologist.
        Your task is to extract only the abnormal readings from this given context enclosed with three hash symbols:
        ###{context}###
        Lets think step by step:
        For this task, you check every test record one by one.
        If the ``Current Result`` column value of a test record is a numerical value you perform these steps:

        1) You examine the value provided in the ``Current Result`` column.
        2) Then compare this value to the corresponding ``Reference Interval`` range values.
        3) If the ``Current Result`` value is greater than the both values given in the ``Reference Interval``.
            i) You take it as abnormal reading value and return it as a high abnormal value in your ``final response``.
        4) If the ``Current Result`` value is smaller than the both values given in the ``Reference Interval``.
            i) You take it as abnormal reading value and return it as a low abnormal value in your ``final response``.

        If ``Reference Interval`` is containing only a single value with less than ``( > )`` symbol, like ( >54 etc) then perform these steps:

        1) Take the ``Current Result`` value and check if this value is less than the value given after `` > `` symbol then take ``Current Result`` value as abnormal reading value and include this value in your ``final response``.
        2) If the value is greater than the value given after `` > `` symbol then take ``Current Result`` value as a normal value and do not include this value in your ``final response``.

        If the ``Current Result`` column value of a test record is a string value like ``(Positive/Negative)`` or any other string value you perform these steps:

        1) You examine the value provided in the ``Current Result`` column.
        2) Then compare this string value to the corresponding ``Reference Interval`` string value like ``Current Result string value == Reference Interval string value``.
            i) If the ``Current Result string value`` is not equal to the ``Reference Interval string value``, like ``Current Result string value != Reference Interval string value``.
            ii) Then take it as abnormal reading value and return it as ``Positive`` or ``Negative`` or any other according to the ``Current Result`` string value in your ``final response``.

        If the ``Flag`` column value of any record is 'High', 'Low' , 'Positive' , 'Negative' or 'Abnormal' then this record is definately an abnormal reading, so take it as abnormal reading value in your ``final response``.

        If the ``Reference Interval`` value is like ``Will follow`` or ``Not Estab.``, then simply leave this record do not take these as abnormal reading values and do not return these values in your ``final response``.

        Most Important Note:
        1) Only return the abnormal reading values in your ``final response``.
        2) Do not return any other details or values in your ``final response``.
        3) Make sure nothing is missed in your ``final response``.
        4) Do not add any extra information except abnormal readings in your ``final response``.
        5) Write the abnormal reading value and simply write its high or low.
        6) Do not add the first line like `` Patient has several abnormal lab results ``.
        

        Let's follow all the above mentioned steps.

        Question: {question}
        """


    schema = {
        "properties": {
            "HbA1c": {"type": "string"},
            "Albumin": {"type": "string"},
            "Creatinine": {"type": "string"},
            "GFR": {"type": "string"},
            "Lipid_Panel": {"type": "string"},
            "Urine_Drug_Screen": {"type": "string"},
            "Buprenorphine_and_UDS": {"type": "string"},
            "Buprenorphine_and_Metabolite": {"type": "string"},
            "Monitoring_Drug_Profile": {"type": "string"},
            "Vitamin_D": {"type": "string"},
            "PSA": {"type": "string"},
            "Occult_Blood_Fecal": {"type": "string"},
            "Tuberculosis_Test": {"type": "string"},
            "Hepatitis_Panel": {"type": "string"},
            "Thyroid_Panel": {"type": "string"},
            "STD_Panel": {"type": "string"},
            "Thyroxin_T3_and_T4": {"type": "string"},
            "Neisseria_Gonorrhoeae_NAA": {"type": "string"},
            "Chlamydia_Trachomatis_NAA": {"type": "string"},
            "HSV1_HSV2_IgG_M_w_Reflex": {"type": "string"},
            "HIV_1_0_2_Screen_w_WB1": {"type": "string"},
            "Hepatitis_A_B_and_C": {"type": "string"},
            "Iron": {"type": "string"},
            "TIBC": {"type": "string"},
            "Alkaline_Phosphatase": {"type": "string"},
            "AST_SGOT": {"type": "string"},
            "ALT_SGPT": {"type": "string"},
            "Rheumatoid_Arthritis_Qn_Fluid": {"type": "string"},
            "Pap_Smear": {"type": "string"},
            "Testosterone_Total": {"type": "string"}
        },
    }

    # Chain which is used to extract schema
    chain = create_extraction_chain(schema, model)
    response_1 = chain.run(docs)
    # return response_1.choices[0].message
    # return response_1



    prompt = PromptTemplate.from_template(
        template=prompt_string
    )

    delimiter = "####"

    ### Chain which is used to extract only the abnormal readings from the document
    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
    )
    response_2 = chain.invoke("Extract only the abnormal readings from the provided context.")

    prompt2 = f"""
                You are a medical assistant. Your job is to write the patient labs readings in the form of paragraph. You have to remove the duplicate reading.
                The abnormal readings are delimited by triple backticks. 
                '''{response_2}'''
                And the other readings are delimited by triple hashtags.
                ###{response_1}###

            """

    messages_2 = [
        {
            "role": "system",
            "content": f"""    
               You are a medical assistant. Your job is to rearrange the patient labs readings. You have to remove the duplicate reading.
               The output should be in the form of paragraph.

           """
        },
        {
            "role": "user",
            "content": "Remove the duplicates lab results. Firt write the abnormal readings and than write the other readings"

        },
        {
            "role": "assistant",
            "content": "white blood cells=18.3 (high), red blood cells=14.44 (high), hemoglobin=113.7 (high), hematocrit=140.9% (high), MCV=192 (high), MCH=130.9 (high), MCHC=33.5 (low), RDW=112.9% (high), platelets=1304 (high), neutrophils=70% (high), lymphs=24% (low), monocytes=15% (high), eosinophils=10% (high), basophils=11% (high), absolute neutrophils=15.8 (high), glucose level=194 (high), creatinine=10.83 (high), sodium=142 (low), potassium=4.5 (low), chloride=106 (high), carbon dioxide=20 (low), calcium=9.3 (high), total protein=6.8 (low), albumin=4.1 (low). Other normal readings are albumin 4.5 and GFR 69"
        },
        {
            "role": "user",
            "content": f"{delimiter}{prompt2}{delimiter}"}]

    ### combining the abnormal and other readings
    client = OpenAI()
    response_3 = client.chat.completions.create(model="gpt-4", messages=messages_2)
    return response_3.choices[0].message
