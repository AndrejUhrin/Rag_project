import streamlit as st
import os
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

load_dotenv()
api_key = os.getenv('GROQ_API_KEY')


st.set_page_config(page_title="Multi-Database Chatbot", layout="wide")
st.session_state.run_timeout = 300  


def initialize_chat():
    return ChatGroq(temperature=0, groq_api_key=api_key, model_name="llama3-70b-8192")


def create_vectorstore(persist_directory):
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-MiniLM-L6-v2')
    return Chroma(persist_directory=persist_directory, embedding_function=embedding_model)

def create_retriever(chat, vectorstore, metadata_field_info, document_content_description):
    retriever = SelfQueryRetriever.from_llm(
        llm=chat,
        vectorstore=vectorstore,
        document_contents=document_content_description,
        metadata_field_info=metadata_field_info,
        verbose=True
    )
    return retriever


def create_qa_chain(chat, retriever):
    custom_prompt_template = """Use the following pieces of information to answer the user's question. Always answear the question as if you were a human and answear in full sentance. During your answear be really specific. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type='stuff',
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa


def handle_query(qa, query):
    result = qa({"query": query})
    return result["result"]


st.title("Multi-Database Chatbot with Streamlit")


tab1, tab2, tab3 = st.tabs(["Article Database", "Entities Database", "Paragraphs Database"])


paths = {
    "Article Database": "../Streamlit_app/article_chroma_db",
    "Entities Database": "../Streamlit_app/entities_chroma_db",
    "Paragraphs Database": "../Streamlit_app/paragraphs_chroma_db"
}


metadata_infos = {
    "Article Database": [
        AttributeInfo(name="article_id", description="Article ID of the paper", type="string"),
        AttributeInfo(name="authors", description="Authors of the paper", type="string or list[string]"),
        AttributeInfo(name="year", description="Year the paper was published", type="integer"),
        AttributeInfo(name="abstract", description="Abstract of the article", type="string"),
        AttributeInfo(name="title", description="Title of the paper", type="string"),
        AttributeInfo(name="keywords", description="Keywords associated with the paper", type="string or list[string]"),
        AttributeInfo(name="citation_count", description="Number of citations the paper has received", type="integer"),
    ],
    "Entities Database": [
        AttributeInfo(name="year", description="Year the paper was published", type="integer"),
        AttributeInfo(name="title", description="Title of the paper", type="string"),
        AttributeInfo(name="last_section_title", description="Title section is associated with paragraph", type="string"),
    ],
    "Paragraphs Database": [
        AttributeInfo(name="year", description="Year the paper was published", type="integer"),
        AttributeInfo(name="title", description="Title of the paper", type="string"),
        AttributeInfo(name="last_section_title", description="Title section is associated with paragraph", type="string"),
    ]
}

document_content_descriptions = {
    "Article Database": "Provides information about article",
    "Entities Database": "Provides context of each paragraph within the article",
    "Paragraphs Database": "Provides context of each paragraph within the article"
}


chats = {name: initialize_chat() for name in paths.keys()}
vectorstores = {name: create_vectorstore(path) for name, path in paths.items()}
retrievers = {name: create_retriever(chats[name], vectorstores[name], metadata_infos[name], document_content_descriptions[name]) for name in paths.keys()}
qas = {name: create_qa_chain(chats[name], retrievers[name]) for name in paths.keys()}


with tab1:
    st.header("Article Database Chatbot")
    query = st.text_input("Enter your query for the Article Database:")
    if st.button("Submit", key="article"):
        answer = handle_query(qas["Article Database"], query)
        st.write("Answer:", answer)

with tab2:
    st.header("Entities Database Chatbot")
    query = st.text_input("Enter your query for the Entities Database:")
    if st.button("Submit", key="entities"):
        answer = handle_query(qas["Entities Database"], query)
        st.write("Answer:", answer)

with tab3:
    st.header("Paragraphs Database Chatbot")
    query = st.text_input("Enter your query for the Paragraphs Database:")
    if st.button("Submit", key="paragraphs"):
        answer = handle_query(qas["Paragraphs Database"], query)
        st.write("Answer:", answer)
