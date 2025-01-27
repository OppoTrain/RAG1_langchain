# RAGgyBot

RAGgyBot is a Retrieval-Augmented Generation (RAG) chatbot built using FastAPI, Streamlit, and AWS services. It combines the power of embeddings, Chroma database retrieval, and Together AI for generating intelligent, context-aware responses. The application allows users to query a knowledge base and receive detailed, human-like answers.

## Features

- **Interactive Chatbot:** Engages users with accurate, contextually relevant responses.
- **RAG Integration:** Utilizes retrieval-augmented generation to combine document retrieval and generative AI.
- **Chroma Database:** Stores and retrieves documents for query resolution.
- **AWS S3 Integration:** Downloads and manages Chroma databases from an S3 bucket.
- **Streamlit Frontend:** A user-friendly web interface for interacting with the chatbot.
- **Dockerized Deployment:** Streamlined deployment using Docker and Docker Compose.

---

## Project Structure

```
RAG1_langchain/
├── app/
│   ├── main.py          # FastAPI backend for processing queries
│   ├── Dockerfile       # Dockerfile for the FastAPI service
├── streamlit/
│   ├── streamlit_app.py # Streamlit frontend for user interaction
│   ├── Dockerfile       # Dockerfile for the Streamlit service
├── docker-compose.yml   # Docker Compose configuration
└── README.md            # Project documentation
```

---

## Requirements

- Python 3.8 or higher
- Docker and Docker Compose
- AWS credentials with access to the specified S3 bucket

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/OppoTrain/RAG1_langchain.git
cd RAG1_langchain
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory and add the following variables:

```env
TOGETHER_API_KEY=<your_together_api_key>
AWS_ACCESS_KEY_ID=<your_aws_access_key>
AWS_SECRET_ACCESS_KEY=<your_aws_secret_key>
AWS_REGION=<your_aws_region>
S3_BUCKET_NAME=<your_s3_bucket_name>
```

### 3. Build and Run Services with Docker Compose

```bash
docker-compose up --build
```

- FastAPI service will run on `http://localhost:8000`
- Streamlit frontend will run on `http://localhost:8501`

---

## Usage

1. Open the Streamlit app at `http://localhost:8501`.
2. Enter your query in the input box and click **Get Answer**.
3. View responses in the chat and refer to the conversation history.

---

## Components

### **FastAPI Backend**
- Hosts the API endpoint `/synthesize/` to process user queries.
- Retrieves documents from Chroma database and generates responses using Together AI.

### **Streamlit Frontend**
- Provides an intuitive web-based interface for user interaction.
- Displays conversation history and allows seamless communication with the backend.

### **Chroma Database**
- Stores embeddings for document retrieval.
- Managed locally and synchronized with AWS S3.

---

## Deployment

1. Ensure AWS credentials are correctly configured for accessing the S3 bucket.
2. Run `docker-compose up` to deploy both the backend and frontend.
3. Access the application via `http://localhost:8501`.

---

## Development

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Backend Locally

```bash
uvicorn app.main:app --reload
```

### Run Frontend Locally

```bash
streamlit run streamlit/streamlit_app.py
```

---

## Technologies Used

- **FastAPI:** Backend framework for API creation.
- **Streamlit:** Web application framework for the frontend.
- **LangChain:** Provides retrieval and embedding functionality.
- **Chroma:** Vector database for storing document embeddings.
- **AWS S3:** Cloud storage for managing Chroma database files.
- **Docker Compose:** Orchestrates multi-container deployment.

---

## Future Enhancements

- Add support for additional retrieval methods (e.g., hybrid retrieval).
- Implement user authentication for personalized sessions.
- Extend the chatbot’s knowledge base with additional data sources.

---

## License

This project is licensed under the [MIT License](LICENSE).

