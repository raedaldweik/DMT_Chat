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
db_path = "risk.db"  # Path to your new SQLite database
engine = create_engine(f"sqlite:///{db_path}")
db = SQLDatabase(engine=engine)

# Step 5: Set Up the LLM Agent
llm = ChatOpenAI(model="gpt-4", temperature=0)
agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

# Step 6: Create a Detailed Data Dictionary (for context)
data_dictionary = """
### Detailed Data Dictionary for 'risk.db'

The database contains five tables: **Risk**, **Action**, **Observation**, **Assessment**, and **Business**.

---

#### 1. Risk
Columns:
- SOURCE_SYSTEM_CD (TEXT)
- RISKINSTANCE_ID (TEXT)
- RISKINSTANCE_RK (BIGINT)
- RISKINSTANCE_NM (TEXT)
- RISKINSTANCE_DESC (TEXT)
- USER_NAME_1 … USER_NAME_5 (FLOAT/TEXT)
- CREATED_USER_NAME (TEXT)
- CREATED_DTTM (TEXT)
- LAST_UPDATE_USER_NAME (FLOAT)
- RISKAPPETITECURRCD (TEXT)
- BUSINESSIMPACTCD & BUSINESSIMPACTCD_C (TEXT)
- OCCURENCEFREQCD & OCCURENCEFREQCD_C (TEXT)
- RISKNATURECD & RISKNATURECD_C (TEXT)
- STATUSCD & STATUSCD_C (TEXT)
- IDENTIFIEDDT (DATETIME)
- NEXTREVIEWDT, RISKAPPETITEAMT, RISKAPPETITEBASEAMT (TEXT)
- RISKSCORE (FLOAT)
- SOXFLG (TEXT)
- CUSTOMUSER1 & CUSTOMUSER2 (TEXT)
- ATTACHMENTS (FLOAT)
- GLB_CUST_OBJ_*_RK & _KEY (TEXT)
- PASBL_RISK, ASMT_RISK (TEXT)
- RISK_ISSUE (FLOAT)
- RISK_CNTRL, RISK_CORESP, PROJECT_RISK, AE_RISK, AUPL_RISK (TEXT)
- RISKINSTANCE_KEY (BIGINT)
- RISKINSTANCE_COUNT (BIGINT)
- LEGAL_ORG_PATH, MANAGEMENT_ORG_PATH, RISK_CAT_PATH (TEXT)
- LEGAL_ORG_RK*, MANAGEMENT_ORG_RK*, RISK_CAT_RK* (TEXT)

#### 2. Action
Columns:
- SOURCE_SYSTEM_CD (TEXT)
- ACTIONPLAN_ID (TEXT)
- ACTIONPLAN_RK (BIGINT)
- ACTIONPLAN_NM, ACTIONPLAN_DESC (TEXT)
- USER_NAME_1 … USER_NAME_5 (TEXT/FLOAT)
- CREATED_USER_NAME, CREATED_FROM_RK (TEXT)
- CREATED_DTTM (DATETIME)
- LAST_UPDATE_USER_NAME (FLOAT)
- AP_STR_MDL_VLDTR_VERIFI (TEXT)
- REFERENCENO (FLOAT)
- ACTIONPLANPRIORITYTYPECD & _C (TEXT)
- STATUSCD & STATUSCD_C (TEXT)
- AP_DTE_CLSD_DT, COMPLETIONDT (TEXT)
- CREATEDDT, ORIGINALTARGETDT, TARGETDT, LATESTREVISEDTARGETDT (DATETIME/TEXT)
- ACTUALCOST (TEXT)
- CUSTOMUSER1 (TEXT)
- GLB_CUST_OBJ_*_RK & _KEY (TEXT)
- ISSUE_AP, AP_AP (TEXT)
- AP_ASMT, CORESP_ACTIONPLAN, POLICY_AP, AUDIT_AP, BUSOBJ_AP, INSPOL_AP, PROINS_AP, ORAP_AP, AP_ESIA, X_*_ACTION, SCR_AP, etc. (FLOAT/TEXT)
- ACTIONPLAN_KEY (BIGINT)
- ACTIONPLAN_COUNT (BIGINT)
- GEOGRAPHY_PATH, MANAGEMENT_ORG_PATH, PROCESS_PATH, PRODUCT_PATH, RESOURCE_DIM_PATH (TEXT)
- GEOGRAPHY_RK*, MANAGEMENT_ORG_RK*, PROCESS_RK*, PRODUCT_RK*, RESOURCE_DIM_RK* (TEXT)

#### 3. Observation
Columns:
- SOURCE_SYSTEM_CD (TEXT)
- KRIOBSERVATION_ID (TEXT)
- KRIOBSERVATION_RK (BIGINT)
- KRIOBSERVATION_NM, KRIOBSERVATION_DESC (TEXT)
- USER_NAME_1 … USER_NAME_5 (FLOAT)
- CREATED_USER_NAME, CREATED_FROM_RK (TEXT)
- CREATED_DTTM (TEXT)
- LAST_UPDATE_USER_NAME (FLOAT)
- JUSTIFICATION, RESPONSESCALERANGE (TEXT)
- KRIFREQUENCYCD & _C, KRINATURECD & _C (TEXT)
- KRITYPECD (FLOAT) & KRITYPECD_C (TEXT)
- SCALETYPECD & _C (TEXT)
- STATUSCD & _C (TEXT)
- UNITOFMEASURECD & _C (TEXT)
- DUEDT, REPORTEDONDT (DATETIME)
- RANGEMAX, RANGEMID1, RANGEMID2, RANGEMIN, SCORE (BIGINT)
- BIGBADFLG, RELAXRANGEFLG (TEXT)
- ATTACHMENTS (FLOAT)
- GLB_CUST_OBJ_*_RK & _KEY (TEXT)
- GLB_PERIOD_RK & _KEY (TEXT)
- LINK_TYPE_RK, LINK_INSTANCE_RK (TEXT)
- BUSINESS_OBJECT_RK_*, BUSINESS_OBJECT_TYPE_NM_* (TEXT)
- KRIOBS_OWNER_1 (TEXT)
- KRIOBSERVATION_COUNT (BIGINT)
- CAUSE_PATH, CONTROL_PATH, LEGAL_ORG_PATH, MANAGEMENT_ORG_PATH, PROCESS_PATH, PRODUCT_PATH, PROJECT_PATH, RESOURCE_DIM_PATH, RISK_CAT_PATH (TEXT)
- CAUSE_RK*, CONTROL_RK*, LEGAL_ORG_RK*, MANAGEMENT_ORG_RK*, PROCESS_RK*, PRODUCT_RK*, PROJECT_RK*, RESOURCE_DIM_RK*, RISK_CAT_RK* (TEXT)

#### 4. Assessment
Columns:
- SOURCE_SYSTEM_CD (TEXT)
- ASSESSMENT_ID (TEXT)
- ASSESSMENT_RK (BIGINT)
- ASSESSMENT_NM, ASSESSMENT_DESC (TEXT)
- USER_NAME_1 … USER_NAME_5 (TEXT/FLOAT)
- CREATED_USER_NAME, CREATED_FROM_RK (TEXT)
- CREATED_DTTM (TEXT)
- LAST_UPDATE_USER_NAME (FLOAT)
- ACTUALSTARTDTTMSTR, RATINGSTEMPLATESTR (TEXT)
- ASBLTYPECD & _C, STAGECD & _C, STATUSCD & _C (TEXT)
- ACTUALENDDT, ACTUALSTARTDT, DUEDT, PLANNEDENDDT, PLANNEDSTARTDT (TEXT)
- AUTORELATEASBLSFLG, RISKDECISIONFLG (TEXT)
- CUSTOMUSER1, CUSTOMUSER2, CUSTOMUSER3 (TEXT)
- ATTACHMENTS (FLOAT)
- GLB_CUST_OBJ_*_RK & _KEY (TEXT)
- AP_ASMT, ASMT_ASMTPD, ASMT_ANSHT, ASMT_PASBL, ASMT_ISSUE, ASMT_CORESP, ASMT_ASMT, ASMT_RISK, ASMT_CNTRL, POLICY_ASMT, CAUSE_ASMT, ASMT_CREREP (FLOAT/TEXT)
- ASSESSMENT_KEY (BIGINT)
- LINK_TYPE_RK, LINK_INSTANCE_RK (TEXT)
- BUSINESS_OBJECT_RK_*, BUSINESS_OBJECT_TYPE_NM_* (TEXT)
- ASMT_ASSESSOR_1 (TEXT)
- ASSESSMENT_COUNT (BIGINT)
- GEOGRAPHY_PATH, MANAGEMENT_ORG_PATH (TEXT)
- GEOGRAPHY_RK*, MANAGEMENT_ORG_RK* (TEXT)

#### 5. Business
Columns:
- SOURCE_SYSTEM_CD (TEXT)
- BUSINESSIMPACTANALYSIS_ID & _RK (BIGINT)
- BUSINESSIMPACTANALYSIS_NM (TEXT)
- BUSINESSIMPACTANALYSIS_DESC (FLOAT)
- USER_NAME_1 … USER_NAME_5 (FLOAT)
- CREATED_USER_NAME, CREATED_FROM_RK (TEXT)
- CREATED_DTTM (TEXT)
- LAST_UPDATE_USER_NAME (FLOAT)
- ALTLOCATION, DEPRECATEDPROCESSES, DISASTEROCCURENCE, PLANACTIVATION, RESUMEPRIMARY, TESTRESULTS (TEXT/FLOAT)
- LOCATIONFLG & _C (TEXT/BIGINT)
- MAO & MAO_C (TEXT)
- P_BUS_24HRCD…P_BUS_8HRCD_C, P_FIN_1WKCD…P_FIN_72HRCD_C, P_REP_1WKCD…P_REP_8HRCD_C (TEXT/FLOAT/BIGINT)
- RPO & RPO_C, RTO & RTO_C (TEXT/BIGINT)
- STATUSCD & STATUSCD_C (TEXT)
- ACTUALENDDT, ACTUALSTARTDT, DUEDT, PLANNEDENDDT, PLANNEDSTARTDT, TESTEDONDT (DATETIME)
- TESTEDFLG (TEXT)
- CUSTOMUSER1, CUSTOMUSER2 (TEXT)
- GLB_CUST_OBJ_*_RK & _KEY (TEXT)
- PROCESS_BIA, BIA_RISKSCENARIO, SCR_BIA (FLOAT/TEXT)
- BUSINESSIMPACTANALYSIS_KEY (BIGINT)
- BUSINESSIMPACTANALYSIS_COUNT (BIGINT)
"""

# Hardcoded Q&A for Executive-Level Queries (you can adjust or extend these)
hardcoded_qa = {
    # e.g. "which risks are highest": "Based on your data, the top risks by score are ...",
}

# Step 7: Build the Streamlit UI
st.title("AI Risk Expert")
st.write("Ask me anything")

# Initialize conversation history in Streamlit session state
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Input field for user query
user_input = st.text_input("You:", key="user_input")

if user_input:
    user_input_lower = user_input.lower().strip()
    query = f"{data_dictionary}\n\n{user_input}"
    try:
        if user_input_lower in hardcoded_qa:
            result = hardcoded_qa[user_input_lower]
        else:
            result = agent_executor.invoke({"input": query})["output"]
        st.session_state.conversation.append(("User", user_input))
        st.session_state.conversation.append(("Assistant", result))
    except Exception as e:
        st.session_state.conversation.append(("Assistant", f"Error: {str(e)}"))
    # Clear input
    st.session_state.user_input = ""

# Display conversation history
for speaker, message in st.session_state.conversation:
    if speaker == "User":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**Assistant:** {message}")
