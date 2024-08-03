import os
import traceback
from dotenv import load_dotenv
import streamlit as st
from langchain.callbacks import get_openai_callback
## Loading Pdfs
def get_agent_executor():
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain.tools.retriever import create_retriever_tool
    
    loader= PyPDFLoader('/Users/vikaslakka/Desktop/FSDS/GenAI/poc/scrum_master/ag_assist/dataset/rm_story.pdf')
    pages= loader.load_and_split()
    ##Create Vector DB with FAISS
    ## Split documents into chunks
    documents= RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    ).split_documents(pages)
    ##Create Vector DB
    vector= FAISS.from_documents(documents, OpenAIEmbeddings())
    retriever= vector.as_retriever()
    ###Create retriever tool
    retriever_tool= create_retriever_tool(
        retriever,
        "story_retrieval",
        "Provides information about story retrieval in Agile project"
    )
    ##Create tool list
    tools= [retriever_tool]
    
    ## Create Agent
    from langchain_openai import ChatOpenAI
    llm= ChatOpenAI(model="gpt-3.5-turbo", temperature=0.6)
    ##Create prompt  for llm agent
    from langchain.prompts import ChatPromptTemplate
    from langchain_core.prompts import MessagesPlaceholder
    template="""
            You are a helpful assistant. This is about retrieving of story details in Agile project.
            Data provided is a story detail. Kindly respond if you find details of the story for the question asked
            otherwise respond politely that you do not know in your tone."""
    story_reader_prompt= ChatPromptTemplate.from_messages(
        [
            ("system", template),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ]
    )
    ##Create agent 
    from langchain.agents import create_tool_calling_agent
    agent= create_tool_calling_agent(llm, tools, story_reader_prompt)
    ## Create Langchain executer
    from langchain.agents import AgentExecutor
    agent_executor= AgentExecutor(agent=agent, tools= tools, verbose=False)
    return agent_executor
load_dotenv()
###Create agent executor
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor=get_agent_executor()
    
##Creating title for application
# st.title("Policy bot: A Knowledge bot which provides you answers regarding policies exist in company")
st.header('Agile Assistant', divider='rainbow')
st.subheader('Assistant for Agile Methodology.')
##Streamlit form creation
with st.form("user_inputs"):

    ask_button= st.form_submit_button("Start discussion")
###########################################
with st.sidebar:
    st.title("Chat here...")
    messages = st.container(height=200)


    if question := st.chat_input("Ask something"):
        messages.chat_message("user").write(question)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})
        ###########################################
        ## Field Validations
        ## if ask_button and upload_file_list is not None and question:len(upload_file_list)>0 and
        if  ask_button is not None and question:

            
            with st.spinner("Fetching details..."):
                try:
                    ##Execute evaluate chain
                    with get_openai_callback() as cb:
                        ## Call agent executor
                        response= st.session_state.agent_executor.invoke({"input": question})
                except Exception as e:
                    traceback.print_exception(type(e),e, e.__traceback__)
                    st.error("Error")      
                
                else:
                    
                    print(f"Total Tokens: {cb.total_tokens}")
                    print(f"Prompt Tokens: {cb.prompt_tokens}")
                    print(f"Completion tokens: {cb.completion_tokens}")
                    print(f"Total Cost:{cb.total_cost}")
                    
                    if isinstance(response, str):
                        answer= response['output']
                        st.text_area(label= "Reponse", value= answer)
                    else:
                        messages.chat_message("assistant").write(f"{response['output']}")
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response['output']})
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Display chat messages from history on app rerun
    if len(st.session_state.messages) >2:
        st.text("Chat-history")
        
        for message in st.session_state.messages[:-2]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])