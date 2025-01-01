import streamlit as st
import requests

st.title("RAGgyBot")
st.header("Welcome to RAGgyBot - Your Intelligent Assistant")

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

def get_response_from_api(user_message):
    try:
        url = "http://3.89.229.219:8000/synthesize/"
        payload = {"question": user_message}
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, json=payload, headers=headers)

        if response.status_code == 200:
            return response.json().get("response", "No response received from the backend.")
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

user_input = st.text_input("Ask RAG:")

if st.button("Get Answer"):
    if user_input:
        st.session_state['messages'].append(f"You: {user_input}")
        bot_response = get_response_from_api(user_input)
        st.session_state['messages'].append(f"RAGgyBot: {bot_response}")
    else:
        st.error("Please enter a valid question.")

st.subheader("Conversation History")
for message in st.session_state['messages']:
    if message.startswith("You:"):
        st.markdown(f"**{message}**")  
    else:
        st.markdown(f"*{message}*")  

st.sidebar.title("Sidebar Options")
option = st.sidebar.selectbox("Choose an option:", ["Home", "About", "Contact"])

if option == "Home":
    st.sidebar.write("You are on the Home page.")
elif option == "About":
    st.sidebar.write("This app demonstrates an interactive chatbot powered by the RAG model.")
elif option == "Contact":
    st.sidebar.write("Contact us at RAGgyBot@gmail.com.")

footer_html = """
    <p style="font-size: 16px; font-weight: bold; background: linear-gradient(to left, #FF6347, #FF4500, #FFD700, #32CD32, #1E90FF); 
    -webkit-background-clip: text; color: transparent;">
    Chatbot powered by knowledge and a dash of magic!
    </p>
"""
st.markdown(footer_html, unsafe_allow_html=True)
