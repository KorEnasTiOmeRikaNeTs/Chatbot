import os
from dotenv import load_dotenv, find_dotenv

import streamlit as st

from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores.pinecone import Pinecone as PineconeLangChain
import pinecone

import utils
from streaming import StreamHandler


st.set_page_config(page_title="ChatPDF", page_icon="📄")
st.header("Chat with your documents")


class CustomDataChatbot:

    def __init__(self):
        self.openai_model = "gpt-3.5-turbo"
        pinecone.Pinecone(
            api_key=os.environ["PINECONE_API_KEY"],
            environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
        )
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.index_name = "langchain-doc-index"
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

    def save_file(self, file):
        folder = "tmp"
        if not os.path.exists(folder):
            os.makedirs(folder)

        file_path = f"./{folder}/{file.name}"
        with open(file_path, "wb") as f:
            f.write(file.getvalue())
        return file_path

    @st.spinner("Analyzing documents..")
    def retrieve_documents(self, uploaded_files):
        docs = []
        for file in uploaded_files:
            file_path = self.save_file(file)
            loader = PyPDFLoader(file_path)
            docs.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)
        PineconeLangChain.from_documents(
            documents=splits, embedding=self.embeddings, index_name=self.index_name
        )

    def qa_chain(self, callbacks):
        vectordb = PineconeLangChain.from_existing_index(
            embedding=self.embeddings, index_name=self.index_name
        )

        retriever = vectordb.as_retriever()

        llm = ChatOpenAI(
            model_name=self.openai_model,
            temperature=0,
            streaming=True,
            callbacks=callbacks,
        )
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm, retriever=retriever, memory=self.memory, verbose=True
        )
        return qa_chain

    @utils.enable_chat_history
    def main(self):

        uploaded_files = st.sidebar.file_uploader(
            label="Upload PDF files", type=["pdf"], accept_multiple_files=True
        )

        if uploaded_files:
            self.retrieve_documents(uploaded_files)

        user_query = st.chat_input(placeholder="Ask me anything!")

        if user_query:
            utils.display_msg(user_query, "user")

            with st.chat_message("assistant"):
                st_cb = StreamHandler(st.empty())
                qa_chain = self.qa_chain(callbacks=[st_cb])
                response = qa_chain.invoke(input={"question": user_query})
                st.session_state.messages.append(
                    {"role": "assistant", "content": response.get("answer")}
                )


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    obj = CustomDataChatbot()
    obj.main()
