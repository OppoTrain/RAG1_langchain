import streamlit as st
import requests

# Title of the application
st.title("RAGgyBot")

# Header
st.header("Welcome to RAGgyBot - Your Intelligent Assistant")

# Initialize session state to store messages if not already initialized
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Function to interact with the FastAPI backend
def get_response_from_api(user_message):
    """
    Sends the user's message to the FastAPI backend and retrieves the chatbot's response.

    Parameters:
    - user_message (str): The user's input message.

    Returns:
    - str: The response from the backend or an error message.
    """
    try:
        url = "http://127.0.0.1:8000/synthesize/"
        payload = {"question": user_message}
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, json=payload, headers=headers)

        if response.status_code == 200:
            return response.json().get("response", "No response received from the backend.")
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Input from the user
user_input = st.text_input("Ask RAG:")

# Button interaction
if st.button("Get Answer"):
    if user_input:
        # Add user's message to the session state
        st.session_state['messages'].append(f"You: {user_input}")

        # Get the chatbot's response from the FastAPI backend
        bot_response = get_response_from_api(user_input)

        # Add chatbot's response to the session state
        st.session_state['messages'].append(f"RAGgyBot: {bot_response}")
    else:
        st.error("Please enter a valid question.")

# Display the chat messages
st.subheader("Conversation History")
for message in st.session_state['messages']:
    if message.startswith("You:"):
        st.markdown(f"**{message}**")  # Format user messages
    else:
        st.markdown(f"*{message}*")  # Format chatbot responses

# Sidebar options for additional features
st.sidebar.title("Sidebar Options")
option = st.sidebar.selectbox("Choose an option:", ["Home", "About", "Contact"])

if option == "Home":
    st.sidebar.write("You are on the Home page.")
elif option == "About":
    st.sidebar.write("This app demonstrates an interactive chatbot powered by the RAG model.")
elif option == "Contact":
    st.sidebar.write("Contact us at RAGgyBot@gmail.com.")

# Footer text with colorful gradient using HTML
footer_html = """
    <p style="font-size: 16px; font-weight: bold; background: linear-gradient(to left, #FF6347, #FF4500, #FFD700, #32CD32, #1E90FF); 
    -webkit-background-clip: text; color: transparent;">
    Chatbot powered by knowledge and a dash of magic!
    </p>
"""
st.markdown(footer_html, unsafe_allow_html=True)
