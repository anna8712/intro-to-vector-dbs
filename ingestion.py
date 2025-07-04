import os 
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter


load_dotenv()

if __name__ == "__main__":
    print("Ingesting...")
    loader = TextLoader("/workspaces/intro-to-vector-dbs/mediumblog1.txt")
    document = loader.load()

    print("Splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    text_documents = text_splitter.split_documents(document)
    print(f"create {len(text_documents)} chunks")
    
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    
    print("ingesting...")
    PineconeVectorStore.from_documents(text_documents, embeddings, index_name = os.environ["INDEX_NAME"])
    print("finish")
    
    