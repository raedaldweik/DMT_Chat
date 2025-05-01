# Step 1: (Optional) Install required packages
# !pip install streamlit langchain langchain_community langchain_openai python-dotenv sqlalchemy

# Step 2: Import Libraries
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
import streamlit as st

# Step 3: Load Environment Variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Please add it to your .env file.")
os.environ["OPENAI_API_KEY"] = api_key

# Step 4: Initialize the SQLite Database
db_path = "risk.db"
engine = create_engine(f"sqlite:///{db_path}")
db = SQLDatabase(engine=engine)

# Step 5: Set Up the LLM Agent (it will introspect your schema automatically)
llm = ChatOpenAI(model="gpt-4", temperature=0)
agent_executor = create_sql_agent(
    llm,
    db=db,
    agent_type="openai-tools",
    verbose=True
)

# Step 6: (Optional) Very-minimal schema hint
# You can omit this entirely if you trust the agent to read the schema.
data_dictionary = "Tables in this database: Risk, Action, Observation, Assessment, Business."

# Step 7: Build the Streamlit UI
st.title("AI Risk Expert")
st.write("Ask me anything about your risk management database")

if "conversation" not in st.session_state:
    st.session_state.conversation = []

user_input = st.text_input("You:", key="user_input")

if user_input:
    # Only pass a one-line schema hint plus the user’s question
    prompt = f"{data_dictionary}\n\n{user_input}"
    try:
        answer = agent_executor.invoke({"input": prompt})["output"]
    except Exception as e:
        answer = f"Error: {str(e)}"
    st.session_state.conversation.append(("You", user_input))
    st.session_state.conversation.append(("Assistant", answer))

for speaker, text in st.session_state.conversation:
    if speaker == "You":
        st.markdown(f"**You:** {text}")
    else:
        st.markdown(f"**Assistant:** {text}")
