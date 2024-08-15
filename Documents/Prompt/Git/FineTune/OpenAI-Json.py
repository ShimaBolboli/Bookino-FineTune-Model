from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
import os
from pinecone import Pinecone, ServerlessSpec
from langchain import PromptTemplate
import streamlit as st
import openai
import logging

# Load environment variables
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai_org_id = os.getenv("OPENAI_ORG_ID")

openai.api_key = openai_api_key
openai.organization = openai_org_id

# Load fine-tuned model ID
with open("fine_tuned_model_id.txt", "r") as f:
    fine_tuned_model_id = f.read().strip()
    print ('/////////////////',fine_tuned_model_id)


# Load and split documents
loader = TextLoader('../book.txt')
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = loader.load_and_split(text_splitter)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize Pinecone instance
pc = Pinecone(api_key=pinecone_api_key, environment='us-east-1')
index_name = "langchain-demo"

# Check if index exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )            
    )

index = pc.Index(index_name)
docsearch = PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)

# Define prompt template
template = """
    You are a helpful assistant who provides book recommendations based on user queries.
    Answer the question in your own words only from the context given to you.
    If questions are asked where there is no relevant context available, please ask the user to ask relevant questions.
    
    Context: {context}
    Question: {question}
    Answer:
"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# Generate a response using the fine-tuned model with RAG
def generate_response(question):
    # Retrieve relevant context from Pinecone
    search_results = docsearch.similarity_search(question, k=3)
    context = "\n".join([result.page_content for result in search_results])
    
 
    # Use the context in the prompt
    formatted_prompt = prompt.format(context=context, question=question)
    logging.info(f"&&&&&&&&&Using model: {fine_tuned_model_id}")
    response = openai.ChatCompletion.create(
        model=fine_tuned_model_id, 

        messages=[
            {"role": "system", "content": "You are a helpful assistant who provides book recommendations based on user queries."},
            {"role": "user", "content": formatted_prompt.strip()}
        ],
        max_tokens=350,
        temperature=0.5
    )
    
    response_text = response['choices'][0]['message']['content'].strip()
    
    # If the response is not relevant or empty, return "No data found"
    if not response_text or "no relevant context" in response_text.lower():
        return "No data found"
    
    return response_text

st.title("📚 Bookino")

# Add subtitle with smaller font
st.markdown("<h4 style='text-align: left; color: gray;'>Book Recommendation Chatbot</h4>", unsafe_allow_html=True)


# Store LLM generated responses
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm here to assist with your book recommendations. How can I help you today?"}]
if "prev_examples" not in st.session_state:
    st.session_state.prev_examples = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if user_input := st.chat_input("Ask your book-related question here"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

# Generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Getting your answer..."):
                response = generate_response(user_input)
                st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})