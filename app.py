from flask import Flask, render_template,jsonify,request
from src.service import download_hugging_face_embeddings
from src.prompt import *
from langchain_pinecone import PineconeVectorStore
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

app=Flask(__name__)

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"]= PINECONE_API_KEY
os.environ["OPENAI_API_KEY"]= OPENAI_API_KEY

embeddings=download_hugging_face_embeddings()

index_name="medicalbot1"

docsearch=PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)

retriever=docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

llm = ChatOpenAI(
    openai_api_key="77d82fc4e93b73b5d222772f81c48afba97e8f1b0d2bf8bfd5e8835f3df3abc9",
    openai_api_base="https://api.together.xyz",
    model_name="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-128K"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])


retrieval_chain = create_stuff_documents_chain(llm, prompt)
rag_chain=create_retrieval_chain(retriever,retrieval_chain)

@app.route("/")
def index():
    return render_template('chat.html')

# @app.route("/get",methods=["GET","POST"])
# def chat():
#     msg=request.form["msg"]
#     input=msg
#     print(input)
#     response=rag_chain.invoke({"input":msg})
#     print("Response : ",response["answer"])
#     return str(response["answer"])
   
@app.route("/", methods=["GET", "POST"])  # Handle both GET and POST on root route
def chat():
    response = None
    if request.method == "POST":
        msg = request.form.get("msg")
        print(f"Received query: {msg}")
        response = rag_chain.invoke({"input": msg})
        response = str(response["answer"])
        print(f"Generated response: {response}")
    
    # Always render the same template, pass response if exists
    return render_template("chat.html", response=response)
if __name__ =='__main__':
    app.run(host="0.0.0.0",port=8080,debug=True)
