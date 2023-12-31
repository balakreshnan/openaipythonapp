# main.py

import os

from flask import (Flask, redirect, render_template, request,
                   send_from_directory, url_for)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
import logging, json, os, urllib
#import azure.functions as func
import openai
from langchain.llms.openai import AzureOpenAI
import os
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
import textwrap
from langchain.document_loaders import AzureBlobStorageFileLoader
from azure.storage.blob import BlobServiceClient, ContainerClient, BlobBlock, BlobClient
from os import environ, path
from dotenv import load_dotenv
from flask import request
import time
import azure.ai.vision as visionsdk

app = Flask(__name__)

@app.route('/')
def index():
   print('Request for index page received')
   return render_template('index.html')

# @app.route('/')
# def index():

#     # Specificy a `.env` file containing key/value config values
#     basedir = path.abspath(path.dirname(__file__))
#     load_dotenv(path.join(basedir, '.env'))
    
#     # General Config
#     OPENAI_API_TYPE = os.environ.get("OPENAI_API_TYPE")
#     OPENAI_API_VERSION = os.environ.get("OPENAI_API_VERSION")
#     OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE")
#     OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
#     DEFAULT_EMBED_BATCH_SIZE = os.environ.get("DEFAULT_EMBED_BATCH_SIZE")
#     BLOB_KEY = os.environ.get("BLOB_KEY")
#     BLOB_URL = os.environ.get("BLOB_URL")
#     OpenAiKey = OPENAI_API_KEY
#     result = "Open AI Api"
#     llm = AzureChatOpenAI(deployment_name="gpt-35-turbo1", model_name="gpt-35-turbo", openai_api_key=OpenAiKey, max_tokens=500)

#     text_splitter = CharacterTextSplitter()

#     account_url = BLOB_URL
#     shared_access_key = BLOB_KEY
#     credential = shared_access_key
#     container_name = "uploaddocs"

#     # Create the BlobServiceClient object
#     blob_service_client = BlobServiceClient(account_url, credential=credential)
#     blob_client = blob_service_client.get_blob_client(container=container_name, blob="samplepdf1.pdf")
#     with open(file='samplepdf2.pdf', mode="wb") as sample_blob:
#         download_stream = blob_client.download_blob()
#         sample_blob.write(download_stream.readall())
#     print('file downloaded')
#     #loader = PyPDFLoader("plexus.pdf")
#     #pages = loader.load_and_split()
#     loader = PyPDFLoader("samplepdf2.pdf")
#     #pages = loader.load_and_split()
#     #len(pages)
#     #faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings(model="text-embedding-ada-002",chunk_size=1))
#     #docs = faiss_index.similarity_search("what is little eletric shocks do?", k=2)
#     #loader = AzureBlobStorageFileLoader(
#     #    conn_str="DefaultEndpointsProtocol=https;AccountName=adbstore;AccountKey=GZf8FqBwlClm8Rkxh2ROH7/SKij9mAcWBxJbRAwUaMk3VEICOFG7UuC3bHGXULnq/HjVKtn3HpDk+ASthMo3Uw==;EndpointSuffix=core.windows.net",
#     #    container="uploaddocs",
#     #    blob_name="samplepdf1.pdf",
#     #)
#     docs = loader.load()
#     prompt_template = """Summarize with bullet items:

#     {text}

#     CONCISE SUMMARY:"""
#     PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
#     chain = load_summarize_chain(llm, chain_type="map_reduce", return_intermediate_steps=False, map_prompt=PROMPT, combine_prompt=PROMPT)
#     result = chain({"input_documents": docs}, return_only_outputs=True)

#     return result

@app.route('/summaryupload', methods=['POST', 'GET'])
def summaryupload():
   print('Request for index page received')
   return render_template('summary.html')

#https://github.com/Azure-Samples/msdocs-python-flask-webapp-quickstart

@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        print('Request for upload page received')
        BLOB_KEY = os.environ.get("BLOB_KEY")
        BLOB_URL = os.environ.get("BLOB_URL")
        f = request.files['file']
        #f.save(secure_filename(f.filename))
        f.save("samplepdf1.pdf")
        account_url = BLOB_URL
        shared_access_key = BLOB_KEY
        credential = shared_access_key
        container_name = "uploaddocs"

        # Create the BlobServiceClient object
        blob_service_client = BlobServiceClient(account_url, credential=credential)
        container_client = blob_service_client.get_container_client(container=container_name)
        with open(file='samplepdf1.pdf', mode="rb") as data:
            blob_client = container_client.upload_blob(name="samplepdf1.pdf", data=data, overwrite=True)
        print('file uploaded successfully')
        return render_template('upload.html')
    else:
        return render_template('upload.html')

@app.route('/summary', methods=['POST', 'GET'])
def summary(filename="samplepdf1.pdf"):
    filename = request.args.get('filename')

        # Specificy a `.env` file containing key/value config values
    basedir = path.abspath(path.dirname(__file__))
    load_dotenv(path.join(basedir, '.env'))
    
    # General Config
    OPENAI_API_TYPE = os.environ.get("OPENAI_API_TYPE")
    OPENAI_API_VERSION = os.environ.get("OPENAI_API_VERSION")
    OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    DEFAULT_EMBED_BATCH_SIZE = os.environ.get("DEFAULT_EMBED_BATCH_SIZE")
    BLOB_KEY = os.environ.get("BLOB_KEY")
    BLOB_URL = os.environ.get("BLOB_URL")
    OpenAiKey = OPENAI_API_KEY
    result = "Open AI Api"
    llm = AzureChatOpenAI(deployment_name="gpt-35-turbo1", model_name="gpt-35-turbo", openai_api_key=OpenAiKey, max_tokens=500)

    #text_splitter = CharacterTextSplitter()
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=3000,chunk_overlap=0,separator="\n\n")

    account_url = BLOB_URL
    shared_access_key = BLOB_KEY
    credential = shared_access_key
    container_name = "uploaddocs"

    # Create the BlobServiceClient object
    blob_service_client = BlobServiceClient(account_url, credential=credential)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob="samplepdf1.pdf")
    with open(file='samplepdf2.pdf', mode="wb") as sample_blob:
        download_stream = blob_client.download_blob()
        sample_blob.write(download_stream.readall())
    print('file downloaded')
    loader = PyPDFLoader("samplepdf2.pdf")
    start = time.time()

    docs = loader.load()
    prompt_template = """Summarize with bullet items:

    {text}

    CONCISE SUMMARY:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = load_summarize_chain(llm, chain_type="map_reduce", return_intermediate_steps=False, map_prompt=PROMPT, combine_prompt=PROMPT)
    result = chain({"input_documents": text_splitter.split_documents(docs)}, return_only_outputs=True)
    end = time.time()

    print(result)
    print(end - start)
    elapsed_time = end - start
    print('Execution time:', elapsed_time, 'seconds')

    #return render_template("upload.html",result = result)
    return result

@app.route('/insights', methods=['POST', 'GET'])
def insights():
   print('Request for index page received')
   return render_template('insights.html')

@app.route('/insightsp', methods=['POST', 'GET'])
def insightsp(filename="samplepdf1.pdf"):
    filename = request.args.get('filename')

        # Specificy a `.env` file containing key/value config values
    basedir = path.abspath(path.dirname(__file__))
    load_dotenv(path.join(basedir, '.env'))
    
    # General Config
    OPENAI_API_TYPE = os.environ.get("OPENAI_API_TYPE")
    OPENAI_API_VERSION = os.environ.get("OPENAI_API_VERSION")
    OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    DEFAULT_EMBED_BATCH_SIZE = os.environ.get("DEFAULT_EMBED_BATCH_SIZE")
    BLOB_KEY = os.environ.get("BLOB_KEY")
    BLOB_URL = os.environ.get("BLOB_URL")
    OpenAiKey = OPENAI_API_KEY
    result = "Open AI Api"
    llm = AzureChatOpenAI(deployment_name="gpt-35-turbo1", model_name="gpt-35-turbo", openai_api_key=OpenAiKey, max_tokens=500)

    #text_splitter = CharacterTextSplitter()
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=3000,chunk_overlap=0,separator="\n\n")

    account_url = BLOB_URL
    shared_access_key = BLOB_KEY
    credential = shared_access_key
    container_name = "uploaddocs"

    # Create the BlobServiceClient object
    blob_service_client = BlobServiceClient(account_url, credential=credential)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob="samplepdf1.pdf")
    with open(file='samplepdf2.pdf', mode="wb") as sample_blob:
        download_stream = blob_client.download_blob()
        sample_blob.write(download_stream.readall())
    print('file downloaded')
    loader = PyPDFLoader("samplepdf2.pdf")
    start = time.time()

    docs = loader.load()
    prompt_template = """Extract Insights with bullet items:

    {text}

    CONCISE SUMMARY:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = load_summarize_chain(llm, chain_type="map_reduce", return_intermediate_steps=False, map_prompt=PROMPT, combine_prompt=PROMPT)
    result = chain({"input_documents": text_splitter.split_documents(docs)}, return_only_outputs=True)

    print(result)
    end = time.time()
    print(end - start)
    elapsed_time = end - start
    print('Execution time:', elapsed_time, 'seconds')

    #return render_template("upload.html",result = result)
    return result

@app.route('/vision', methods=['POST', 'GET'])
def visionupload():
   print('Request for index page received')
   return render_template('vision.html')

@app.route('/imageprocessing', methods=['POST', 'GET'])
def imageprocessing():
    if request.method == 'POST':
        print('Request for upload page received')
        BLOB_KEY = os.environ.get("BLOB_KEY")
        BLOB_URL = os.environ.get("BLOB_URL")
        visionendpoint = os.environ.get("visionendpoint")
        visionkey = os.environ.get("visionkey")
        f = request.files['file']
        #f.save(secure_filename(f.filename))
        f.save("image1.png")
        account_url = BLOB_URL
        shared_access_key = BLOB_KEY
        credential = shared_access_key
        container_name = "uploadimage"
        result = ""

        # Create the BlobServiceClient object
        blob_service_client = BlobServiceClient(account_url, credential=credential)
        container_client = blob_service_client.get_container_client(container=container_name)
        with open(file='image1.png', mode="rb") as data:
            blob_client = container_client.upload_blob(name="image1.png", data=data, overwrite=True)
        print('file uploaded successfully')
        service_options = visionsdk.VisionServiceOptions(visionendpoint, visionkey)
        vision_source = visionsdk.VisionSource(filename="image1.png")
        analysis_options = visionsdk.ImageAnalysisOptions()
        analysis_options.features = (
            visionsdk.ImageAnalysisFeature.CROP_SUGGESTIONS |
            visionsdk.ImageAnalysisFeature.CAPTION |
            visionsdk.ImageAnalysisFeature.DENSE_CAPTIONS |
            visionsdk.ImageAnalysisFeature.OBJECTS |
            visionsdk.ImageAnalysisFeature.PEOPLE |
            visionsdk.ImageAnalysisFeature.TEXT |
            visionsdk.ImageAnalysisFeature.TAGS
        )
        #analysis_options.cropping_aspect_ratios = [0.9, 1.33]
        analysis_options.language = "en"
        analysis_options.model_version = "latest"
        analysis_options.gender_neutral_caption = True
        # Create the image analyzer object
        image_analyzer = visionsdk.ImageAnalyzer(service_options, vision_source, analysis_options)
        print()
        print(" Please wait for image analysis results...")
        print()
        result1 = image_analyzer.analyze()
        result_details = visionsdk.ImageAnalysisResultDetails.from_result(result1)
        result = "image analysis result:"
        print(result_details)
        #result = "Image Analysis Result:"
        #if result.reason == visionsdk.ImageAnalysisResultReason.ANALYZED:
        #    result = result1
        print(" Result details:")
        print("   Image ID: {}".format(result_details.image_id))
        print("   Result ID: {}".format(result_details.result_id))
        print("   Connection URL: {}".format(result_details.connection_url))
        print("   JSON result: {}".format(result_details.json_result))    
        result = result_details.json_result 

        return result
    else:
        return render_template('vision.html', result = "Error in processing image")


@app.route('/hello', methods=['POST'])
def hello():
   name = request.form.get('name')

   if name:
       print('Request for hello page received with name=%s' % name)
       return render_template('hello.html', name = name)
   else:
       print('Request for hello page received with no name or blank name -- redirecting')
       return redirect(url_for('index'))

if __name__ == "__main__":
    app.run()