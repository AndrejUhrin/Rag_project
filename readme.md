In this project, we have outlined multiple approaches to create a RAG. All these approaches are documented in this folder. Here, we will explain which approach was used for the final validation and which ones were unsuccessful.

The final RAG used for evaluation is stored in a folder called [RAG_3_vectordb_3_separate_codes](RAG_3_vectordb_3_separate_codes). This folder contains the following files:

- **chroma_db_3_tables_creation:** Used for creating the vector databases.
- **article_chroma_db:** A vector database containing information about articles from the papers table in DuckDB.
- **entities_chroma_db:** A vector database containing information about entities and sentences from the entities table in DuckDB.
- **paragraphs_chroma_db:** A vector database containing information about paragraphs and section titles from the paragraphs table in DuckDB.
- **RAG_selquerying_articles:** A RAG that uses article_chroma_db for retrieving information.
- **RAG_selquerying_entities:** A RAG that uses entities_chroma_db for retrieving information.
- **RAG_selquerying_paragraphs:** A RAG that uses paragraphs_chroma_db for retrieving information.

## How to Run This Code

1. If the folder does not yet contain the vector databases (article_chroma_db, entities_chroma_db, paragraphs_chroma_db), you need to run the `chroma_db_3_tables_creation` code first. This code creates the three vector databases and stores them in the folder `RAG_3_vectordb_3_separate_codes`. If this folder already contains these vector databases, you can skip this step.
2. Once the vector databases are created, you can run these notebooks: `RAG_selquerying_articles`, `RAG_selquerying_entities`, `RAG_selquerying_paragraphs`.
   - These codes use the Groq API key for the LLM model, which you need to obtain from the Groq website (it's free). Store this key in a `.env` file. You can get this API key by following this tutorial: [Groq API Key Tutorial](https://www.youtube.com/watch?v=VmNhDUKMHd4&list=LL&index=1)
3. If you want to ask a new question to RAG, simply create a new cell following this format:
   ```python
   query = "here input your question"
   result = qa({"query": query})
   print("Answer:", result["result"])

Additional code related to the folder `RAG_3_vectordb_3_separate_codes` is in the folder `Streamlit app`, which serves a similar function but can be run locally as an interface.

## Folders with Previous Approaches

There are also a couple of folders containing our previous attempts and approaches to creating a RAG. Here, you need to follow the same steps as previously mentioned. These approaches are not included in the final evaluation in the report.

- **[RAG_identical_metadata_page_content](RAG_identical_metadata_page_content):** In this approach, we attempted to create a single vector database containing all the necessary information for answering our questions. Ultimately, it performed quite okay for qualitative questions but poorly for quantitative questions.
  
- **[RAG_multiple_vector_stores](RAG_multiple_vector_stores):** In this approach, we tried to combine all vector databases into one question-answering chain. However, we were unable to successfully implement this solution. We encountered errors when asking quantitative questions and received incorrect answers for qualitative questions.

- **[First Attempt RAG](first_attempt_rag):** This was our initial approach and test to understand how to build a RAG. It is created on a very small subset and does not contain any information for evaluation.



