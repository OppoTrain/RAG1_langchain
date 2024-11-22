from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_together import Together
from dotenv import load_dotenv
import re
import os
import boto3
# Disable oneDNN optimizations in TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load environment variables from the .env file
load_dotenv()

# Access the environment variable for API key
together_key = os.getenv('TOGETHER_API_KEY')

# Initialize FastAPI
app = FastAPI()

# AWS S3 Credentials and Bucket Information
aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')  # Add your Access Key ID in .env
aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')  # Add your Secret Access Key in .env
region = os.getenv('AWS_REGION', 'eu-north-1')  # Default to Stockholm region if not provided
bucket_name = os.getenv('S3_BUCKET_NAME', 'alaasbucket')  # Bucket name
s3_folder_path = 'chroma/'  # Folder path in S3

# Local folder to download the Chroma DB
local_folder_path = "chroma_local/"
chroma_db_file = "chroma.sqlite3"
chroma_db_path = os.path.join(local_folder_path, chroma_db_file)

# Ensure the folder exists
if not os.path.exists(local_folder_path):
    os.makedirs(local_folder_path)

# Initialize the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Initialize the S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name=region
)

# List and download Chroma files from S3
def download_chroma_from_s3():
    try:
        # List objects in the S3 folder
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=s3_folder_path)

        # Download each file
        for obj in response.get("Contents", []):
            s3_file_key = obj["Key"]
            local_file_path = os.path.join(local_folder_path, os.path.relpath(s3_file_key, s3_folder_path))
            
            # Create local directories if necessary
            local_dir = os.path.dirname(local_file_path)
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)

            # Download the file if it does not already exist
            if not os.path.exists(local_file_path):
                print(f"Downloading {s3_file_key} to {local_file_path}")
                s3_client.download_file(bucket_name, s3_file_key, local_file_path)

                # Check if the file was downloaded successfully
                if os.path.exists(local_file_path):
                    print(f"Successfully downloaded {s3_file_key} to {local_file_path}")
                else:
                    print(f"Failed to download {s3_file_key}")
            else:
                print(f"File {local_file_path} already exists. Skipping download.")

    except Exception as e:
        print(f"Error while downloading Chroma from S3: {e}")

# Check if the Chroma database exists locally
def ensure_chroma_exists():
    if not os.path.exists(chroma_db_path):
        # If the Chroma database is not found, download it from S3
        print(f"Chroma database not found at {chroma_db_path}. Downloading from S3.")
        download_chroma_from_s3()
        
        # After download, check if the file exists
        if not os.path.exists(chroma_db_path):
            raise FileNotFoundError(f"Chroma database file not found at {chroma_db_path}. Ensure it is downloaded properly.")
    else:
        print(f"Chroma database already exists at {chroma_db_path}, using the existing database.")

# Update the persist directory to target only the directory
persist_directory = local_folder_path  # Set to 'chroma_local/'
chroma_db = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)


# Define request model for the query
class Query(BaseModel):
    question: str

# Define function to clean and format the answers
def clean_and_format_answers(answers):
    clean_html = re.compile('<.*?>')
    cleaned_text = ""
    for answer in answers:
        cleaned_text += f"Question: {answer['question']}\n\n"
        for doc in answer['relevant_documents']:
            cleaned_doc = re.sub(clean_html, '', doc)  # Clean HTML tags
            cleaned_text += f"{cleaned_doc.strip()}\n\n"
    return cleaned_text.strip()

# Endpoint for retrieving answers from Chroma
@app.post("/retrieve/")  # Example: POST /retrieve/
async def retrieve_answer(query: Query):
    try:
        retriever = chroma_db.as_retriever(search_type="similarity", k=4, relevance_score_threshold=0.55)
        results = retriever.get_relevant_documents(query.question)
        relevant_docs = [result.page_content for result in results]
        answers = [{"question": query.question, "relevant_documents": relevant_docs}]
        cleaned_text = clean_and_format_answers(answers)
        return {"retrieved_text": cleaned_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint for synthesizing a response using the LLM
@app.post("/synthesize/")  # Example: POST /synthesize/
async def synthesize_response(query: Query):
    try:
        llm = Together(
            model="meta-llama/Llama-2-13b-chat-hf",
            together_api_key=together_key,
            temperature=0.1,
            max_tokens=800  # Explicitly set max_tokens
        )
        retriever = chroma_db.as_retriever(search_type="similarity", k=4, relevance_score_threshold=0.55)
        results = retriever.get_relevant_documents(query.question)
        relevant_docs = [result.page_content for result in results]
        answers = [{"question": query.question, "relevant_documents": relevant_docs}]
        cleaned_text = clean_and_format_answers(answers)

        # Debug: Print the cleaned_text
        print("Cleaned Text:", cleaned_text)

        # Define the prompt for the LLM
        prompt = PromptTemplate(
            input_variables=["history", "input"],
            template = """
                This is a conversation between a human and an AI assistant familiar with human rights.
                {history}
                Human: I will provide a text from the retrieved documents and the question asked. Please formulate a coherent response based on the information provided.
                Be sure to highlight all important aspects of human rights mentioned in the text.
                If specific articles or laws related to human rights are mentioned in the text, please refer to them explicitly.
                In addition, be neutral in any response and make your primary reference the retrieved documents that I will send you.
                If sources are available, please refer to the documents appropriately.
                Text from the retrieved documents and the question asked:
                {input}
                AI:"""
        )
        history = ""
        formatted_prompt = prompt.format(history=history, input=cleaned_text)
        response = llm.invoke(formatted_prompt)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))