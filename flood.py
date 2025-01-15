# Step 1: (Optional) Install required packages
# !pip install streamlit langchain langchain_community langchain_openai python-dotenv sqlalchemy

# Step 2: Import Libraries
import os
import sqlite3
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
db_path = "flood.db"  # Path to your SQLite database
engine = create_engine(f"sqlite:///{db_path}")
db = SQLDatabase(engine=engine)

# Step 5: Set Up the LLM Agent
# Make sure your model name is available in your environment (e.g., "gpt-4" or "gpt-3.5-turbo")
llm = ChatOpenAI(model="gpt-4", temperature=0)
agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

# Step 6: Create a Detailed Data Dictionary (for context)
data_dictionary = """
### Detailed Data Dictionary for 'flood.db'

The database contains three tables: **inundation_forecasting**, **rainfall**, and **alerts**.
Below is a detailed breakdown of each column and their relationships.

---

#### 1. inundation_forecasting
- **datetime**: Timestamp indicating the date and time of the forecast (e.g. "2025-01-01 08:00:00").
- **ht_forecast**: Numerical value representing the predicted water height or inundation level (e.g. "5.2 feet").
- **latitude**: Geographic latitude coordinate for the forecast location (e.g. "24.4539").
- **longitude**: Geographic longitude coordinate for the forecast location (e.g. "54.3773").

**Purpose**:  
This table is used to store predictive data for flooding. The `ht_forecast` helps determine flood severity, and the lat/long fields map exactly where the forecast is applicable.

---

#### 2. rainfall
- **precipInches**: Amount of rainfall measured in inches (e.g. "0.75").
- **deviceId**: Unique identifier of the device recording rainfall (e.g. "R123").
- **deviceLabel**: Human-readable label/name of the device (e.g. "Downtown Rain Gauge").
- **region**: Region or area name associated with the device (e.g. "Abu Dhabi City Center").
- **msr_date**: Date of the measurement (e.g. "2025-01-05").
- **msr_time**: Time of the measurement (e.g. "14:30:00").
- **latitude**: Geographic latitude coordinate of the device (e.g. "24.4539").
- **longitude**: Geographic longitude coordinate of the device (e.g. "54.3773").
- **msr_dttm**: Combined date-time of the measurement, often used for queries and analysis (e.g. "2025-01-05 14:30:00").

**Purpose**:  
This table tracks rainfall data from various devices. The `region` column can be linked to the region in alerts or used in conjunction with lat/long to match forecasting data or alerts.

---

#### 3. alerts
- **alertType**: Type of alert (e.g. "Flood Warning", "Flash Flood", "High Tide").
- **alertSubtype**: More specific subtype of the alert (e.g. "Severe", "Moderate").
- **alertSeverity**: Severity level (e.g. "Critical", "High", "Medium", "Low").
- **alertMessage**: Text describing the alert (e.g. "Flash Flood Watch in effect until 6 PM.").
- **deviceId**: Device ID associated with the alert; may link to `rainfall.deviceId` if relevant.
- **region**: Region name where the alert is applicable (e.g. "Al Nahdah").
- **msr_dttm**: Combined date-time when the alert was issued (e.g. "2025-01-05 14:45:00").
- **msr_date**: Date when the alert was issued (e.g. "2025-01-05").
- **msr_time**: Time when the alert was issued (e.g. "14:45:00").
- **msr_month**: Month of the alert, either numeric or textual (e.g. "January" or "01").
- **Lat**: Latitude coordinate of the alert location (e.g. "24.4539").
- **Long**: Longitude coordinate of the alert location (e.g. "54.3773").

**Purpose**:  
This table stores alerts that have been triggered, including the type of flood alert, severity, and location details.

---

### Potential Relationships:
1. **alerts.deviceId** ⇔ **rainfall.deviceId**:  
   If both the `alerts` and `rainfall` tables share the same device ID, they can be joined to correlate which specific rainfall device triggered an alert.
2. **alerts.region** ⇔ **rainfall.region**:  
   If you want to track all alerts for a given region, you can join on the `region` column. The same can be done with `inundation_forecasting` if you assign region names consistently or rely on the lat/long proximity to match the region.
3. **Spatial/Geographic Matching**:  
   By comparing lat/long values in `inundation_forecasting`, `rainfall`, and `alerts`, you can determine how different meteorological or hydrological events line up in the same area.

---

### Usage:
Use this data dictionary to understand the purpose of each field when querying the database or building flood-related analytics.
"""

# Hardcoded Q&A for Executive-Level Queries
hardcoded_qa = {
    "which area will have a high impact for future floods": (
        "Based on our historical data and current forecasting models, "
        "the areas that exhibit the highest vulnerability to future flood events "
        "are **Al Adlah**, **Al Nahdah**, **Bu Deeb**, and **Al Haffar**. "
        "These locations consistently appear in our inundation forecasts due to "
        "their geographical profiles and proximity to low-lying flood plains."
    ),
    "recommendation to reduce impact": (
        "To mitigate the risk in these high-impact areas, we recommend an integrated approach:\n\n"
        "1. **Infrastructure Upgrades**: Enhance and maintain drainage systems, and consider building "
        "   protective levees or flood barriers.\n"
        "2. **Smart Monitoring**: Install additional rainfall gauges and flood sensors for real-time monitoring.\n"
        "3. **Urban Planning**: Implement zoning regulations to limit construction in flood-prone zones.\n"
        "4. **Community Preparedness**: Conduct regular flood drills, ensure early-warning systems are in place, "
        "   and provide public education on emergency response."
    ),
    "why are these areas impacted": (
        "These regions are particularly vulnerable due to a combination of factors:\n\n"
        "- **Topography**: Areas like Al Adlah and Al Nahdah have lower elevations, causing water to accumulate.\n"
        "- **Coastal Proximity**: Bu Deeb is near a coastal plain, making it susceptible to storm surges.\n"
        "- **Drainage and Infrastructure**: Al Haffar’s drainage systems may require updates to handle heavy rainfall.\n"
        "- **Historical Patterns**: Data shows these areas have experienced recurring flood incidents, "
        "   indicating underlying vulnerabilities that require focused intervention."
    ),
}

# Step 7: Build the Streamlit UI
st.title("Flood Digital Assistant")
st.write("Ask me anything about floods in Abu Dhabi")

# Initialize conversation history in Streamlit session state
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Input field for user query
user_input = st.text_input("You:", key="user_input")

if user_input:
    # Convert input to lowercase for exact dictionary matching
    user_input_lower = user_input.lower().strip()

    # Combine data dictionary with user query for better context
    query = f"{data_dictionary}\n\n{user_input}"

    try:
        # Check if user_input is one of our hardcoded questions (exact match in lowercase)
        if user_input_lower in hardcoded_qa:
            result = hardcoded_qa[user_input_lower]
        else:
            # If not in hardcoded Q&A, use the LLM agent to handle the query
            result = agent_executor.invoke({"input": query})["output"]

        st.session_state.conversation.append(("User", user_input))
        st.session_state.conversation.append(("Assistant", result))
    except Exception as e:
        st.session_state.conversation.append(("Assistant", f"Error: {str(e)}"))

    # Clear the input field after submission
    user_input = ""

# Display conversation history
for speaker, message in st.session_state.conversation:
    if speaker == "User":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**Assistant:** {message}")
