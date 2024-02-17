import os
from dotenv import load_dotenv, find_dotenv

import whisper
import streamlit as st
from langchain import hub
from langchain.agents import AgentExecutor, load_tools, create_react_agent
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory

import utils


st.set_page_config(page_title="Chatbot", page_icon="💬")
st.header("Audio-transcription Chatbot")


class Basic:

    def __init__(self):
        self.openai_model = "gpt-3.5-turbo"

    def save_file(self, file):
        folder = "tmp"
        if not os.path.exists(folder):
            os.makedirs(folder)

        file_path = f"./{folder}/{file.name}"
        with open(file_path, "wb") as f:
            f.write(file.getvalue())
        return file_path

    @st.spinner("Analyzing audio..")
    def get_user_query_from_mp3(self, uploaded_files):
        whisper_model = whisper.load_model("medium")
        user_query = []
        for file in uploaded_files:
            file_path = self.save_file(file)
            user_query.append(
                utils.get_transcript(file_name=file_path, model=whisper_model)
            )
        return "".join(user_query)

    def setup_agent(self, callbacks):
        llm = ChatOpenAI(
            temperature=0.4,
            model_name=self.openai_model,
            streaming=True,
            callbacks=callbacks,
        )
        prompts = hub.pull("hwchase17/react")
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="input",
            output_key="output",
        )
        tools = load_tools(["serpapi", "llm-math", "wikipedia"], llm=llm)
        agent = create_react_agent(llm=llm, tools=tools, prompt=prompts)
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            handle_parsing_errors=True,
            verbouse=False,
        )
        return executor

    @utils.enable_chat_history
    def main(self):

        uploaded_files = st.sidebar.file_uploader(
            label="Upload audio files", type=["ogg", "mp3"], accept_multiple_files=True
        )
        user_query = st.chat_input(placeholder="Ask me anything!")

        if uploaded_files:
            user_query = self.get_user_query_from_mp3(uploaded_files=uploaded_files)

        if user_query:
            utils.display_msg(user_query, "user")

            with st.chat_message("assistant"):
                st_cb = StreamlitCallbackHandler(st.container())
                agent = self.setup_agent(callbacks=[st_cb])
                output = agent.invoke(input={"input": user_query})
                st.session_state.messages.append(
                    {"role": "assistant", "content": output.get("output")}
                )
                st.write(output.get("output"))


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    obj = Basic()
    obj.main()
