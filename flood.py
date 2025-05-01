# Step 1: (Optional) Install required packages
# !pip install streamlit langchain langchain_community langchain_openai python-dotenv sqlalchemy

# Step 2: Import Libraries
import os
import pandas as pd
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
db_path = "risk.db"  # <-- switched to your risk.db
engine = create_engine(f"sqlite:///{db_path}")
db = SQLDatabase(engine=engine)

# Step 5: Set Up the LLM Agent
llm = ChatOpenAI(model="gpt-4", temperature=0)
agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

# Step 6: Create a Condensed Data Dictionary to avoid context limits
data_dictionary = """
### risk.db schema (condensed)

**Tables (5):**
1. **Risk** – risk instances  
   - ~70 columns: IDs, names, descriptions, scores, appetite & impact codes, status, timestamps, org paths, custom fields.

2. **Action** – action plans  
   - ~115 columns: IDs, names, descriptions, priority, status, dates, costs, lookup keys, org/process/product paths.

3. **Observation** – KRI observations  
   - ~85 columns: IDs, names, descriptions, scales & frequencies, due/report dates, range values, flags, org & KRI links.

4. **Assessment** – control/risk assessments  
   - ~80 columns: IDs, names, descriptions, stages, statuses, start/end dates, flags, scores, org & object links.

5. **Business** – business impact analyses  
   - ~90 columns: IDs, names, descriptions, impact metrics (24 hr–72 hr, RPO/RTO), statuses, dates, flags, org & BIA links.
"""

# Optional: exact-match Q&A overrides
hardcoded_qa = {
    # e.g. "list all risks": "Here are the risk instances..."
}

# Step 7: Build the Streamlit UI
st.title("Risk Digital Assistant")
st.write("Ask me anything!")

# Initialize conversation history
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Text input widget
user_input = st.text_input("You:", key="user_input")

if user_input:
    # Combine condensed schema + user question
    query = f"{data_dictionary}\n\n{user_input}"
    try:
        # Check for hardcoded Q&A
        key = user_input.lower().strip()
        if key in hardcoded_qa:
            result = hardcoded_qa[key]
        else:
            result = agent_executor.invoke({"input": query})["output"]
    except Exception as e:
        result = f"Error: {str(e)}"

    # Append to session history
    st.session_state.conversation.append(("You", user_input))
    st.session_state.conversation.append(("Assistant", result))

# Display conversation history
for speaker, message in st.session_state.conversation:
    if speaker == "You":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**Assistant:** {message}")
