# Step 1: (Optional) Install required packages
# pip install streamlit langchain langchain-community langchain-openai python-dotenv sqlalchemy openai

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
import streamlit as st
import pandas as pd

# Step 2: Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY not found. Please add it to your .env file.")
    st.stop()
os.environ["OPENAI_API_KEY"] = api_key

# Step 3: Initialize the SQLite database
db_path = "risk.db"
engine = create_engine(f"sqlite:///{db_path}")

# If you need to load CSV data into the DB, uncomment and adjust:
# df = pd.read_csv("your_risk_data.csv")
# df.to_sql("Risk", engine, if_exists="replace", index=False)
# (Repeat for other tables: Action, Observation, Assessment, Business)

db = SQLDatabase(
    engine=engine,
    include_tables=["Risk", "Action", "Observation", "Assessment", "Business"],
    sample_rows_in_table_info=False,
)

# Step 4: Set up the LLM agent
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.0)
agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    agent_type="openai-tools",
    verbose=False,
)

# Optional one‐line schema hint
schema_hint = (
    "This database has five tables: Risk, Action, Observation, Assessment, Business.\n"
    "Feel free to reference columns and join tables as needed."
)

# Step 5: Build the Streamlit UI
st.set_page_config(page_title="AI Risk Expert", page_icon="🛡️")
st.title("🛡️ AI Risk Expert")
st.write("Ask me anything about your risk management database!")

if "conversation" not in st.session_state:
    st.session_state.conversation = []

user_input = st.text_input("You:", key="user_input")
if user_input:
    prompt = f"{schema_hint}\n\nUser question: {user_input}"
    try:
        # Depending on your LangChain version, use .run() or .invoke()
        answer = agent_executor.run(prompt)
    except Exception as e:
        answer = f"Error: {e}"
    st.session_state.conversation.append(("You", user_input))
    st.session_state.conversation.append(("Assistant", answer))

# Display chat history
for speaker, text in st.session_state.conversation:
    if speaker == "You":
        st.markdown(f"**You:** {text}")
    else:
        st.markdown(f"**Assistant:** {text}")
