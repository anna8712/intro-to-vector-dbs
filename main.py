import os
from dotenv import load_dotenv

from langchain import hub
from langchain_community.document_loaders import TextLoader

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

from langchain_core.prompts import PromptTemplate

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


if __name__ == "__main__":

    load_dotenv()

    print("Retrieving...")

    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI()

    query = "what is Pincecone in machine learning?"

    vectore_store = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(
        llm,
        retrieval_qa_chat_prompt,
    )
    retrieval_chain = create_retrieval_chain(
        retriever=vectore_store.as_retriever(),
        combine_docs_chain=combine_docs_chain,
    )

    result = retrieval_chain.invoke(input={"input": query})

    print(result)
