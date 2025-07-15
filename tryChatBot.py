from datetime import datetime
import os
from click import prompt
from langsmith import expect
import streamlit as st
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferMemory 
from langchain_openai import ChatOpenAI, OpenAI


def initialize_session_state():
    if 'chat_history'  not in st.session_state:
        st.session_state.chat_history = []
    if 'total_messages'  not in st.session_state:
        st.session_state.total_messages = 0
    if 'start_time'  not in st.session_state:
        st.session_state.start_time = None

def get_custom_prompt():
    """Get custom prompt template based on persona"""
    persona = st.session_state.get('selected_persona','Default')

    personas = {
        'Default' : """You are a helpful AI assitance.
                    Current conversation:
                    {history}
                    Human: {input}
                    AI:""",
        'Expert' : """You are an expert consultant and deep knowladge across multiple fields.
                    Please provide detail and technical responses when appropriate.
                    Current conversation:
                    {history}
                    Human: {input}
                    Expert:""",
        'Creative' : """You are an crative and imaginative AI that thinks outside the box.
                    Feel free to use metaphors and analogies in your responses.
                    Current conversation:
                    {history}
                    Human: {input}
                    Creative:""",
    }

    return PromptTemplate(
        input_variables=["history","input"],
        template=personas[persona]
    )

# API key
OAK = os.environ["OPEN_API_KEY"] = "gsk-proj-uvTWArLCijMjFNVXRHjMvxSG5cuN6jmwzfhc5SRlqAuZ0bzb7pPTGhgwM6SdkK1QS0iatexY75T3BlbkFJVTXR-zbledjlk_6-cSOlN2FzG_3_Q3bEA7-9lvpN-18luuVzwqsVVDjZrOIyhpxeCoImoCWKcA"

# set page layout
st.set_page_config(
    page_title="OpenAI Chat Application",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

initialize_session_state()

# Sidebar configuraation
with st.sidebar:
    st.title("Chat settings")

    # Choose a model
    st.subheader("Model selection")
    model = st.sidebar.selectbox(
        'Choose a model',
        ['gpt-3.5-turbo', 'gpt-4', 'gpt-4o'],
        help="Select the AI model for your conversation"
    )

    # Save Chat history
    st.subheader("Memory settings")
    memory_length = st.sidebar.slider("Convertion memory (messages)",1, 10, 5,
                    help="Number of previous messages to remember")


    # Persona Selection
    st.subheader("AI Persona")
    st.session_state.selected_persona = st.selectbox(
        'Select Converstion Style:',
        ['Default','Expert','Creative']
    )

    #Chat statistics
    if st.session_state.start_time:
        st.subheader("Chat Statisics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages",len(st.session_state.chat_history))
        with col2:
            duration = datetime.now() - st.session_state.start_time
            st.metric("Duration",f"{duration.seconds // 60}m {duration.seconds}")

    #Clear chat button
    if st.button("Clear Chat History",use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.start_time = None
        st.rerun()
    
# Give a title
st.title("OpenAI Chat Assistant")

#Memory
memory = ConversationBufferMemory(k = memory_length)

# Open AI Chat Model 
llm = ChatOpenAI(
    openai_api_key = OAK,
    model_name = model
)

# Chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=get_custom_prompt()
)

#Load chat history from memory
for message in st.session_state.chat_history:
    memory.save_context(
        {'input':message['human']},
        {'output':message['AI']}
    )

# Display chat history
for message in st.session_state.chat_history:
    # user message
    with st.container():
        st.write('You')
        st.info(message['human'])
    
    with st.container():
        st.write(f"'Assistant'({st.session_state.selected_persona}mode)")
        st.success(message['AI'])

# Add Some Spacing
st.write("")

#User input section
st.markdown("### Your message")
user_question = st.text_area(
    "",
    height=100,
    placeholder="Type your message here (Shift + Enter to send)",
    key="user_input",
    help="Type your message and press Shift + Enter or click the send button"
    )

# Input buttons
col1,col2,col3 = st.columns([3,1,1])
with col2:
    send_button = st.button("Send",use_container_width=True)
with col3:
    if st.button("New Topic",use_container_width=True):
        memory.clear()
        st.success("Memory cleard for new topic!")

if send_button and user_question:
    if not st.session_state.start_time:
        st.session_state.start_time = datetime.now()    

with st.spinner("Thinking..."):
    try:
        response=conversation(user_question)
        manage = {
            'human':user_question,
            'AI':response['response']
        }
        st.session_state.chat_history.append(message)
        st.rerun()
    except Exception as e:
        st.error(f"Error:{str(e)}")

#Footer 
st.markdown("---")
st.markdown(
    "Using OpenAI with "
    f"{st.session_state.selected_persona.lower()} persona | "
    f"Memory : {memory_length} messages"
)

    




