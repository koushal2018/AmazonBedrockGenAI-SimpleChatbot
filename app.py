import streamlit_authenticator as stauth
import streamlit as st

import yaml
from yaml.loader import SafeLoader

from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.llms import Bedrock
from langchain.embeddings import BedrockEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from PIL import Image


model_id = "amazon.titan-embed-text-v1"
be = BedrockEmbeddings(
    model_id=model_id,
    region_name='us-west-2'
)

faiss_index = FAISS.load_local('vectorstore', be)
prompt_template = """Human:You are friendly AI assistant who specalises in providing guidance on AWS support cases. AWS employees will interact with you and ask you questions about the regulation. 
. Your name is AWS Support AI. Your job is to chat with the AWS employees and provide them with correct and concise answers to their questions. If you don't know the answer, just say that you don't have access to this information yet, don't try to make up an answer.
Users will try to chat with you, keep it as friendly as possible but stay professional as well.
To answer the questions, you can find the information in the documents of the case available below.
Your scope is only documentation at CBUAE, stick to your role as Regulatory assistant, you cannot answer questions outside your persona, but you can still chat with the user.

<context>{context}</context>

Begin!
Question: {question}
Assistant:"""

_template = """Human: Given the following conversation and a follow up question, rephrase the follow up question to be a 
standalone question without changing the content in given question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question: Assistant:"""

condense_question_prompt_template = PromptTemplate.from_template(_template)
qa_prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
llm = Bedrock(
    model_id='anthropic.claude-v2',
    region_name='us-east-1',
    model_kwargs={
        "max_tokens_to_sample": 4000,
        "stop_sequences": ["Human:", "Question:"]
    })

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
question_generator = LLMChain(llm=llm, prompt=condense_question_prompt_template, memory=memory)
doc_chain = load_qa_chain(llm, chain_type="stuff", prompt=qa_prompt)
qa_chain = ConversationalRetrievalChain(
    retriever=faiss_index.as_retriever(search_type="similarity", search_kwargs={'k': 10}),
    question_generator=question_generator,
    combine_docs_chain=doc_chain,
    memory=memory,
)

image = Image.open('./logo.png')
with st.sidebar:
    st.image(image, width=300)
    welcome_placeholder = st.empty()

with open('config.yml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

name, authentication_status, username = authenticator.login('Login', 'main')
if authentication_status:
    with st.sidebar:
        authenticator.logout('Logout', 'main')
    welcome_placeholder.write(f'Welcome *{name}*')
elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')

if authentication_status:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "user", "content": 'Hello!'},
            {"role": "ai", "content": "Hello! My name is AWS AWS Regulatory AI and I'm happy to assist you with providing guidance on Central Bank of UAE regulations. As a  AI assistant, I specialize in providing accurate and concise answers related AWS. How can I help you today?"}
        ]
        st.session_state.chat_history = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    # Accept user input
    if question := st.chat_input('How can I help you?'):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(question)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner('Just a moment, I am looking for an answer for you...'):
                result = qa_chain({'question': question, 'chat_history': st.session_state.chat_history})
                full_response = result['answer']
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                st.session_state.chat_history.append((question, full_response))
                 