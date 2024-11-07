import os
import openai
import gradio as gr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


# Directly passing the API key to the OpenAI model
api_key="YOUR OPEN AI KEY HERE"

os.environ['OPENAI_API_KEY'] = api_key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize the OpenAI model using ChatOpenAI (with GPT-4 or the desired model)
llm = ChatOpenAI(model_name="gpt-4", temperature=0)  # Update the model name if needed

# Function to load and display content from all PDF files in the folder
def load_documents_from_folder(folder_path):
    # List to store the content of all documents
    documents_content = {}
    
    # Iterate over each file in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is a PDF
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            
            # Load the document using PyPDFLoader
            loader = PyPDFLoader(file_path)
            document = loader.load()  # This will return the document content
            
            # Store the content of the document in the dictionary
            documents_content[filename] = document
    
    # Return the dictionary with document names and their contents
    return documents_content

# Folder path where the PDF files are located
folder_path = r'C:\Users\hp\Desktop\Gen AI\PWC'

# Load documents from the folder
documents = load_documents_from_folder(folder_path)

# Split documents into chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(documents)

# Create embeddings and store in FAISS
def create_vector_store(document_chunks):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(document_chunks, embeddings)


# User access control (admin and end-user roles)
user_access_control = {
    "admin": {
        "username": "admin",
        "password": "adminpassword",
        "access": ["file1.pdf", "file2.pdf", "file3.pdf"]
    },
    "end-user": {
        "username": "user",
        "password": "userpassword",
        "access": ["file1.pdf", "file2.pdf"]
    }
}

# Function to authenticate user
def authenticate_user(username, password):
    for role, user_data in user_access_control.items():
        if user_data['username'] == username and user_data['password'] == password:
            return role, user_data['access']
    return None, []

# Function for user login
def login(username, password):
    role, accessible_files = authenticate_user(username, password)
    if role:
        return (
            f"Login successful. Welcome {role}!",
            role,
            accessible_files,
            gr.update(visible=True)  # Show Query Document tab on success
        )
    return "Invalid username or password. Please try again.", None, [], gr.update(visible=False)

# Function to handle query submission after login
def query_document(query, role, accessible_files):
    if role is None:
        return "You must log in first."
    
    print(f"Received query: {query}")  # Debug log
    
    if query:
        # Load documents based on user access
        folder_path = r'C:\Users\hp\Desktop\Gen AI\PWC'  # Folder where PDFs are stored
        documents = load_documents_from_folder(folder_path, accessible_files)
        
        if not documents:
            return "No accessible documents found."
        
        # Split the documents into chunks for further processing
        document_chunks = split_documents(documents)
        
        # Create vector store from the document chunks
        vector_store = create_vector_store(document_chunks)
        
        # Using MMR for retrieval
        retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 5})
        
        # Retrieve relevant documents for the query
        retrieved_docs = retriever.invoke(query)
        
        if not retrieved_docs:
            return "No relevant documents found."

        # Generate a response using the retrieved documents
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        prompt = f"Given the following context, answer the question:\n\n{context}\n\nQuestion: {query}"

        # OpenAI model response
        response = llm(prompt)
        return response
    return "No query entered."

# Build the Gradio UI
def build_ui():
    with gr.Blocks() as demo:
        # Login Tab
        with gr.Tab("Login") as login_tab:
            gr.Markdown("**Login Form**")
            username = gr.Textbox(label="Username")
            password = gr.Textbox(label="Password", type="password")
            login_button = gr.Button("Login")
            output_login = gr.Textbox(label="Login Status")
            role, accessible_files = gr.State(), gr.State()
            
            # Query Document Tab - Initially hidden
            with gr.Tab("Query Document", visible=False) as query_tab:
                gr.Markdown("**Query the Document Knowledge Base**")
                query_input = gr.Textbox(label="Enter your query:")
                output_query = gr.Textbox(label="Query Response")
                submit_query = gr.Button("Submit Query")
                
                # Process the query submission after login
                submit_query.click(
                    fn=query_document, 
                    inputs=[query_input, role, accessible_files], 
                    outputs=[output_query]
                )

            # Login button logic for tab visibility
            login_button.click(
                fn=login,
                inputs=[username, password],
                outputs=[output_login, role, accessible_files, query_tab]
            )

    demo.launch(share=True)

# Run the UI
if __name__ == "__main__":
    build_ui()