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
from langchain.text_splitter import CharacterTextSplitter

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

load_dotenv()

together_key = os.getenv('TOGETHER_API_KEY')

app = FastAPI()

aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')  
aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')  
region = os.getenv('AWS_REGION', 'eu-north-1')  
bucket_name = os.getenv('S3_BUCKET_NAME', 'alaasbucket')  
s3_folder_path = "chroma/"  
download_directory = "chroma_local/"  

if not os.path.exists(download_directory):
    os.makedirs(download_directory)

s3_client = boto3.client(
    's3',
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name=region
)

def download_chroma_from_s3():
    """
    Downloads all files from the specified S3 folder to the local directory.
    Ensures files are downloaded only if they do not exist or are incomplete.
    """
    try:
        
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=s3_folder_path)
        
        if "Contents" not in response or len(response["Contents"]) == 0:
            print("No files found in the specified S3 folder.")
            return
        
        print(f"Found {len(response['Contents'])} objects in S3 under the path: {s3_folder_path}")

        for obj in response["Contents"]:
            s3_file_key = obj["Key"]
            if s3_file_key.endswith("/"):
                
                continue

            
            local_file_path = os.path.join(download_directory, os.path.relpath(s3_file_key, s3_folder_path))

            local_dir = os.path.dirname(local_file_path)
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)

            if not os.path.exists(local_file_path) or os.path.getsize(local_file_path) != obj["Size"]:
                print(f"Downloading {s3_file_key} to {local_file_path}...")
                s3_client.download_file(bucket_name, s3_file_key, local_file_path)

                if os.path.exists(local_file_path) and os.path.getsize(local_file_path) == obj["Size"]:
                    print(f"Successfully downloaded {s3_file_key}.")
                else:
                    print(f"Failed to download {s3_file_key}. File may be incomplete.")
            else:
                print(f"File {local_file_path} already exists and is up-to-date. Skipping download.")
    except Exception as e:
        print(f"Error while downloading from S3: {e}")


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

chroma_db_file = "chroma.sqlite3"
chroma_db_path = os.path.join(download_directory, chroma_db_file)

def ensure_chroma_exists():
    if not os.path.exists(chroma_db_path):
        print(f"Chroma database not found at {chroma_db_path}. Downloading from S3.")
        download_chroma_from_s3()
        
        if not os.path.exists(chroma_db_path):
            raise FileNotFoundError(f"Chroma database file not found at {chroma_db_path}. Ensure it is downloaded properly.")
    else:
        print(f"Chroma database already exists at {chroma_db_path}, using the existing database.")

ensure_chroma_exists()

persist_directory = download_directory  
chroma_db = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)

class Query(BaseModel):
    question: str

def clean_and_format_answers(answers):
    clean_html = re.compile('<.*?>')
    cleaned_text = ""
    
    for answer in answers:
        for i, doc in enumerate(answer['relevant_documents']):
            if isinstance(doc, str):
                cleaned_doc = re.sub(clean_html, '', doc)
                cleaned_doc = cleaned_doc.strip()
                
                keywords = answer['question'].lower().split()
                if any(keyword in cleaned_doc.lower() for keyword in keywords):
                    cleaned_text += f"Document {i+1}:\n{cleaned_doc}\n\n"
    
    return cleaned_text.strip()

def create_retriever(chroma_db, search_type="similarity", threshold=0.55, k=4, lambda_mult=0.25):
    retriever = chroma_db.as_retriever(
        search_type=search_type,
        relevance_score_threshold=threshold,
        k=k,
        lambda_mult=lambda_mult
    )
    return retriever

def truncate_and_format_context(cleaned_text, max_length=3000):
    """
    Truncates and formats the context to a manageable length while preserving the most relevant information.
    """

    sentences = cleaned_text.split('\n')
    formatted_text = []
    current_length = 0
    
    for sentence in sentences:
        if current_length + len(sentence) <= max_length:
            formatted_text.append(sentence)
            current_length += len(sentence)
        else:
            break
            
    return '\n'.join(formatted_text)

@app.post("/synthesize/")
async def synthesize_response(query: Query):
    try:
        llm = Together(
            model="meta-llama/Llama-2-13b-chat-hf",
            together_api_key=together_key,
            temperature=0.3,
            max_tokens=3000,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1
        )

        retriever = create_retriever(chroma_db, search_type="similarity", threshold=0.55, k=4, lambda_mult=0.25)
        results = retriever.get_relevant_documents(query.question)

        relevant_docs = [result.page_content for result in results]
        
        if relevant_docs:
            answers = [{"question": query.question, "relevant_documents": relevant_docs}]
            cleaned_text = clean_and_format_answers(answers)

            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template="""Based on the question and the document received, provide a comprehensive and complete answer.
                Context:
                {context}
                Question: {question}
                Provide a focused answer using only the information from the given context.
                AI:
                """
            )
            
            formatted_prompt = prompt.format(
                context=cleaned_text,
                question=query.question
            )
        else:
            
            formatted_prompt = f"""
            The question is: "{query.question}"
            Unfortunately, I couldn't find any reference material related to your query.
            However, based on general knowledge, here is an attempt to answer your question:
            """

        print(f"Formatted Prompt: {formatted_prompt}")
        print(f"Context length: {len(formatted_prompt)}")
        
        response = llm.invoke(formatted_prompt)
        
        if not response or len(response.strip()) < 5:
            return {"response": "The model returned an insufficient response. Please try again."}
            
        return {"response": response}
    
    except Exception as e:
        print(f"Error in synthesize_response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
