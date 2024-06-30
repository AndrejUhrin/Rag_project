import os
import streamlit as st
import duckdb
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from langchain_groq import ChatGroq

# Initialize the ChatGroq client with API key from environment variable
api_key = os.getenv('GROQ_API_KEY')
chat = ChatGroq(temperature=0, groq_api_key=api_key, model_name="llama3-70b-8192")

# Constants for database paths
CHROMA_DB_PATH = "/Users/kiri/Documents/HWR/master/SS/Text & Web/Rag_project-1/first attempt rag/chroma_db/chroma.sqlite3"
DUCKDB_PATH = "/Users/kiri/Documents/HWR/master/SS/Text & Web/Rag_project-1/duck_db"

# Function to load the Chroma DB
def load_chroma_db(db_path):
    settings = Settings(persist_directory=db_path, chroma_db_impl="sqlite")
    client = chromadb.PersistentClient(settings=settings)
    return client

# Function to query the DuckDB
def query_duckdb(db_path, query):
    conn = duckdb.connect(database=db_path, read_only=True)
    result = conn.execute(query).fetchall()
    conn.close()
    return result

# Function to get an answer from the RAG system using ChatGroq API
def get_rag_answer(prompt, chroma_client, duckdb_path):
    # Implement your RAG logic using both the Chroma client and DuckDB
    embeddings = embedding_functions.get_embedding(prompt)
    search_results = chroma_client.query(embeddings, top_k=5)
    
    # Assume we get document IDs from search results, then query DuckDB for details
    document_ids = [result['document_id'] for result in search_results['documents']]
    documents = query_duckdb(duckdb_path, f"SELECT * FROM documents WHERE id IN {tuple(document_ids)}")

    # Prepare documents as context
    context = "\n\n".join([str(doc) for doc in documents])

    # Use ChatGroq API to generate a response based on documents and prompt
    response = chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"{prompt}\n\n{context}",
            }
        ]
    )

    return response.choices[0].message.content

# Streamlit UI
def main():
    # Use HTML and CSS for positioning the headers
    st.markdown("""
        <div style="display: flex; justify-content: space-between;">
            <div style="font-size: large; font-weight: bold;">Business School of Economics and Law</div>
            <div style="font-size: large; font-weight: bold;">Text, Web and Social Media Analytics Lab</div>
        </div>
        <hr style="margin-top: 10px; margin-bottom: 10px;">
    """, unsafe_allow_html=True)

    st.title("Streamlit app for RAG Project")
    prompt = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        chroma_client = load_chroma_db(CHROMA_DB_PATH)
        answer = get_rag_answer(prompt, chroma_client, DUCKDB_PATH)
        st.write("Answer:", answer)

    # Add the names at the bottom
    st.markdown("***")
    st.markdown("Developed by: Andrej Uhrin, Yasmin Hilles, Kira Duske, Lydia Wong")

if __name__ == "__main__":
    main()
