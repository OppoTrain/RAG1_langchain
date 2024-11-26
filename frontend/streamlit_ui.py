import streamlit as st

# Title of the application
st.title("RAGgyBot")

# Header
st.header("Welcome to RAGgyBot")

# Initialize session state to store messages if not already initialized
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Function to simulate chatbot response
def get_chatbot_response(user_message):
    # Convert user input to lowercase for easier comparison
    user_message = user_message.lower()
    
    # Respond based on the user's input
    if "hi" in user_message or "hello" in user_message:
        return "RAGgyBot: Hello! How can I assist you today?"
    elif "bye" in user_message:
        return "RAGgyBot: Goodbye! Have a great day!"
    else:
        return f"RAGgyBot: Here's your response!"  # Default response

# Input from the user
user_input = st.text_input("Ask RAG:")

# Button interaction
if st.button("Get Answer"):
    if user_input:
        # Add user's message to the session state
        st.session_state['messages'].append(f"You: {user_input}")
        
        # Get the chatbot's response
        bot_response = get_chatbot_response(user_input)
        
        # Add chatbot's response to the session state
        st.session_state['messages'].append(bot_response)
        
    else:
        st.error("What’s on your mind?")

# Display the chat messages
for message in st.session_state['messages']:
    if message.startswith("You:"):
        st.markdown(f"**{message}**")  # User message
    else:
        st.markdown(f"*{message}*")  # Chatbot response

# Sidebar example
st.sidebar.title("Sidebar Options")
option = st.sidebar.selectbox("Choose an option:", ["Home", "About", "Contact"])

if option == "Home":
    st.sidebar.write("You are on the Home page.")
elif option == "About":
    st.sidebar.write("This app showcases an interactive chatbot powered by the RAG model for intelligent responses.")
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
