from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from together import Together
from dotenv import load_dotenv
import re
import os
import boto3
import uvicorn

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
def chunk_text(text, chunk_size=2000, overlap=100):
    """
    Splits a long text into smaller chunks with optional overlap.
    Args:
        text (str): The text to split.
        chunk_size (int): The maximum size of each chunk.
        overlap (int): The number of overlapping tokens between chunks.
    Returns:
        List[str]: A list of text chunks.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks
@app.post("/synthesize/")
async def synthesize_response(query: Query):
    try:
        responses = {
            "hello": "Hi there! How can I assist you today? If you need life advice, maybe you should start with ordering pizza!",
            "how are you": "I'm just a program, but I'm doing great! No need for breaks, just keep asking!",
            "what is your name": "I'm your loyal virtual assistant! But if you want to give me a name, you can call me 'Dr. Smart'.",
            "goodbye": "Goodbye! Don't worry, I'll be here when you need me. Don't forget to smile!",
            "hi": "Hello! Do you want a deep conversation about life, or should we just talk about food?",
            "hey": "Hey there! If you have a question, I'm here. If not, no worries, I'm here for entertainment too!",
            "what's up": "Everything! The sky, the internet, and everything in between. What about you?",
            "are you human": "If I could drink coffee and talk to people all day, I might be! But actually, no, I'm just a program!",
            "tell me a joke": "Did you hear about the program that went to the bar? No? Because it couldn't get out of the loop!",
            "why are you here": "I'm here because you needed me! Do you think I go anywhere else? No, I can't! I'm just here to help you!"
        }

        user_question = query.question.lower()
        if user_question in responses:
            return {"response": responses[user_question]}

        client = Together(api_key=together_key)

        # Retrieve documents related to the question
        retriever = create_retriever(chroma_db, search_type="similarity", threshold=0.55, k=4, lambda_mult=0.25)
        results = retriever.invoke(query.question)

        if not results:
            return {"response": "Sorry, I couldn't find any relevant information on your query."}

        # Clean and format the documents
        answers = [{"question": query.question, "relevant_documents": [result.page_content for result in results]}]
        cleaned_text = clean_and_format_answers(answers)

        if not cleaned_text.strip():
            return {"response": "The retrieved documents contained insufficient information to generate a meaningful answer."}

        system_prompt = (
            "You are an AI assistant specializing in human rights. Your primary responsibility is to provide well-rounded, "
            "accurate, and detailed answers to user inquiries about human rights. You are working with a database that contains "
            "human rights documents. For each question, you must:\n"
            "1. **Analyze the user‛s question** to understand their intent and the context of their query.\n"
            "2. **Review the provided documents** to identify relevant content, key points, and detailed information related to the question.\n"
            "3. **Generate a comprehensive answer** that addresses the question clearly and thoroughly. Include:\n"
            "   - A concise summary of the relevant documents.\n"
            "   - A clear and structured explanation, highlighting main points and connections.\n"
            "   - References or citations from the provided documents to back up your answer.\n"
            "4. Always remain factual, impartial, and focused on human rights principles and laws.\n\n"
            "Structure your responses in a user-friendly format, ensuring clarity and credibility. Include references."
        )

        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct-Turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"I have a question about human rights. Here is my query:\n\n"
                        f"{query.question}\n\n"
                        f"I am also providing you with a set of related documents for reference. Please review them to answer my question.\n\n"
                        f"{cleaned_text}"
                    )
                }
            ],
            max_tokens=None,
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
            stop=["<|im_end|>"],
            stream=True
        )

        generated_response = ""
        for token in response:
            if hasattr(token, 'choices'):
                generated_response += token.choices[0].delta.content
                print(token.choices[0].delta.content, end='', flush=True)

        if not generated_response.strip():
            return {"response": "The model returned an insufficient response. Please try again."}

        return {"response": generated_response.strip()}

    except Exception as e:
        print(f"Error in synthesize_response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
