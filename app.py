import os
import streamlit as st
from neo4j import GraphDatabase
import json
import re
from datetime import datetime, timedelta
import traceback
import google.generativeai as genai
import ast # Import the ast module for literal_eval
from typing import Union # Import Union for type hinting

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# Ragas Imports
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from datasets import Dataset
from ollama import Client as OllamaClient
import ollama
ollama_client = OllamaClient()

# --- Configuration Import and Validation ---
# Initialize config variables with None as defaults. This ensures they always exist
# in the global scope, even if they are missing from config.py.
NEO4J_URI = None
NEO4J_USERNAME = None
NEO4J_PASSWORD = None
GEMINI_API_TOKEN = None
RAG_LLM_MODEL = None
RAGAS_LLM_MODEL = None # Ensure this is explicitly initialized

try:
    # Attempt to import specific variables from config.py
    # This will overwrite the None defaults if the variables exist in config.py
    from config import (
        NEO4J_URI,
        NEO4J_USERNAME,
        NEO4J_PASSWORD,
        GEMINI_API_TOKEN,
        RAG_LLM_MODEL,
        RAGAS_LLM_MODEL,
        STUDENT_MODEL
    )
    # After importing, check if any critical variables are still None or empty
    missing_vars = []
    if not NEO4J_URI: missing_vars.append("NEO4J_URI")
    if not NEO4J_USERNAME: missing_vars.append("NEO4J_USERNAME")
    if not NEO4J_PASSWORD: missing_vars.append("NEO4J_PASSWORD")
    if not GEMINI_API_TOKEN: missing_vars.append("GEMINI_API_TOKEN")
    if not RAG_LLM_MODEL: missing_vars.append("RAG_LLM_MODEL")
    if not RAGAS_LLM_MODEL: missing_vars.append("RAGAS_LLM_MODEL")
    if not STUDENT_MODEL: missing_vars.append("STUDENT_MODEL")

    if missing_vars:
        st.error(f"Configuration error: The following variables are missing or empty in 'config.py': {', '.join(missing_vars)}. Please define them correctly.")
        # Set a flag to indicate critical config is missing
        config_ok = False
    else:
        config_ok = True

except ImportError:
    st.error("Error: 'config.py' not found. Please ensure it exists in the same directory as 'omg.py'.")
    config_ok = False
except Exception as e:
    st.error(f"An error occurred during config import: {e}. Please check config.py syntax.")
    traceback.print_exc()
    config_ok = False


# --- Config Setup ---
# Only proceed with API setup if all critical configuration variables were successfully loaded and are not empty
if config_ok:
    genai.configure(api_key=GEMINI_API_TOKEN)
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        # Test connection by running a simple query
        with driver.session() as session:
            session.run("RETURN 1").single()
    except Exception as e:
        st.error(f"Failed to connect to Neo4j database with provided credentials: {e}. Please check NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD in config.py.")
        driver = None # Set driver to None if connection fails
        config_ok = False # Mark config as not OK if DB connection fails

    # Initialize LLM clients only if API token and model names are available
    if GEMINI_API_TOKEN and RAG_LLM_MODEL:
        client = genai.GenerativeModel(RAG_LLM_MODEL)
        agent_llm = ChatGoogleGenerativeAI(model=RAG_LLM_MODEL, google_api_key=GEMINI_API_TOKEN, temperature=0.0)
        # Set GOOGLE_API_KEY environment variable for Ragas and other tools that might implicitly use it
        os.environ["GOOGLE_API_KEY"] = GEMINI_API_TOKEN
    else:
        st.error("Gemini API token or RAG_LLM_MODEL is missing. LLM functionalities will be disabled.")
        client = None
        agent_llm = None
        config_ok = False # Mark config as not OK

else:
    st.warning("Critical configuration variables are missing or invalid. Most functionalities will be disabled.")
    driver = None
    client = None
    agent_llm = None


# --- Streamlit Chatbot UI ---
st.title("📊 OntoMetrics ChatBot")

# --- Initialization ---
# Ensure session state variables are initialized only once
if "greeting_shown" not in st.session_state:
    st.session_state.greeting_shown = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "evaluation_data" not in st.session_state:
    st.session_state.evaluation_data = [] # Stores data for Ragas evaluation
if "show_metrics_for_msg" not in st.session_state:
    st.session_state.show_metrics_for_msg = {} # Stores which message's metrics should be shown
if "show_cypher_for_msg" not in st.session_state:
    st.session_state.show_cypher_for_msg = {} # Stores which message's cypher query should be shown

# --- Greeting Logic ---
# Display the greeting only if it hasn't been shown yet in the current session
if not st.session_state.greeting_shown:
    with st.chat_message("assistant"):
        st.write("Hello there! I'm your AI assistant.I can provide reasoning on various Business Metrics in Consumer Wireline,Wireless areas. How can I help you today?.\n\n  Start conversation by typing **Hi** or **Help** to know what data i can support.")
    st.session_state.greeting_shown = True # Mark as shown

# --- Ragas Evaluation Function ---
def run_ragas_evaluation():
    if not st.session_state.evaluation_data:
        st.warning("No data available to run Ragas evaluation. Ask a question first!")
        return

    st.write("**Ragas Evaluation Results**")

    questions = [item["question"] for item in st.session_state.evaluation_data]
    answers = [item["answer"] for item in st.session_state.evaluation_data]
    contexts = [item["contexts"] for item in st.session_state.evaluation_data]
    ground_truths = [item.get("ground_truth", "") for item in st.session_state.evaluation_data] # IMPORTANT: Retrieve ground_truth

    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths, # IMPORTANT: Pass ground_truth to dataset
    }

    try:
        dataset = Dataset.from_dict(data)
        st.write("Ragas Dataset created successfully.")

        # Define the metrics to evaluate
        metrics = [
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ]

        st.write("Running Ragas evaluation... This might take a moment.")
        with st.spinner("Calculating Ragas metrics..."):
            # Check if RAGAS_LLM_MODEL and GEMINI_API_TOKEN are available and valid
            if not RAGAS_LLM_MODEL or not GEMINI_API_TOKEN: # <--- Updated check
                st.error("Ragas LLM model name or Gemini API token is missing or invalid. Please check your config.py.")
                return

            ragas_llm = ChatGoogleGenerativeAI(model=RAGAS_LLM_MODEL, google_api_key=GEMINI_API_TOKEN, temperature=0.0)
            ragas_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_TOKEN)

            result = evaluate(
                dataset,
                metrics=metrics,
                llm=ragas_llm, # Pass the Gemini LLM
                embeddings=ragas_embeddings # Pass the Gemini Embeddings
            )
        st.write("Ragas evaluation complete!")
        st.dataframe(result.to_pandas())

    except Exception as e:
        st.error(f"Error during Ragas evaluation: {e}")
        st.info("Make sure your `GOOGLE_API_KEY` is correctly set in your environment or `config.py` and `RAGAS_LLM_MODEL` is defined.")
        st.warning("Some Ragas metrics (like context_recall, context_precision) require `ground_truth` in the dataset to be fully meaningful. If you don't have human-labeled ground truths, these metrics might show NaNs or unexpected values.")
        traceback.print_exc()


# Display previous messages using st.chat_message
for idx, msg in enumerate(st.session_state.chat_history):
    if msg["role"] == "user":
        with st.chat_message("user"): # This aligns to the left
            st.markdown(f"<span style='color:#0000FF;'>{msg['content']}</span>", unsafe_allow_html=True) # Blue color
    else:
        with st.chat_message("assistant"): # This aligns to the right
            st.markdown(f"<span style='color:#ff2500;'>{msg['content']}</span>", unsafe_allow_html=True) # Green color
            # If the assistant message has a cypher query, display the expander
            if "cypher_query" in msg and msg["cypher_query"]: # This condition ensures expander only shows if there's a query
                with st.expander("More Info"):
                    # Button to show Cypher Query
                    if st.button("Show Generated Cypher Query", key=f"show_cypher_button_{idx}"):
                        st.session_state.show_cypher_for_msg[idx] = not st.session_state.show_cypher_for_msg.get(idx, False)
                        for k in list(st.session_state.show_cypher_for_msg.keys()):
                            if k != idx:
                                st.session_state.show_cypher_for_msg[k] = False
                        # Removed st.rerun() here as it can cause issues during processing of new queries

                    # Display Cypher Query if the button for this message was pressed
                    if st.session_state.show_cypher_for_msg.get(idx, False):
                        st.write("**Generated Cypher Query:**")
                        st.code(msg["cypher_query"], language="cypher")

                    # Button to show metrics for this specific message
                    if st.button("Show Cypher Query Metrics", key=f"show_metrics_button_{idx}"):
                        # Toggle the state for this message
                        st.session_state.show_metrics_for_msg[idx] = not st.session_state.show_metrics_for_msg.get(idx, False)
                        # Ensure only one metrics view is open at a time if desired, otherwise remove this loop
                        for k in list(st.session_state.show_metrics_for_msg.keys()):
                            if k != idx:
                                # Fix typo: st.session_session -> st.session_state
                                st.session_state.show_metrics_for_msg[k] = False
                        # Removed st.rerun() here as it can cause issues during processing of new queries

                    # Display metrics if the button for this message was pressed
                    if st.session_state.show_metrics_for_msg.get(idx, False):
                        if "cypher_metrics" in msg and msg["cypher_metrics"]:
                            st.write("**Cypher Query Traversal / Metadata Metrics:**")
                            for metric_name, metric_value in msg["cypher_metrics"].items():
                                st.write(f"{metric_name.replace('_', ' ').title()}: {metric_value}")
                        else:
                            st.info("No query traversal metrics available for this query.")

                    # Existing Ragas button
                    if st.button("Show Ragas Evaluation Metrics", key=f"ragas_button_{idx}"):
                        run_ragas_evaluation()


# Input at bottom
user_query = st.chat_input("Ask a question about telecom KPIs:")

# --- Function: Extract entities & relationship ---
def get_relevant_metrics_and_relationships(query: str, metric_list) -> dict:
    if not client:
        st.error("LLM client not initialized. Cannot extract entities.")
        return {}
    prompt = f"""
    Given the user question: "{query}", extract:
    1. Metrics mentioned or implied. available metrics are {', '.join(metric_list)}.
    2. Type of relationship (e.g., positive, negative, influenced)
    3. product type for example wireless,wireline etc. if no product type mentioned then take "All Products".
    4. service for example for wireline voice, video , data and for wireless product type prepaid or postpaid.. do not keep empty string list.
    5. US region or major market area if mentioned, e.g., southeast , west , new york,California, Texas, etc. if we can not determine region then use "All Regions".
    6. Time period eg given month , last month,quarter , q1,q2,q3,q4,full year, last year, etc.
        1st Quarter means q1, 2nd Quarter means qq2, 3rd Quarter means q3, 4th Quarter means q4.
        if two quaters are mentioned then use q1,q2 or q3,q4 etc.
        Do not include year in the time period column.
    7. year if mentioned, eg 2024 , 2025 , "last year", "rest of the year","current year","this year" etc. if not mentioned then keep it empty.
    8. if the user query is asking for general performance or scorecard and did not mention any metric name then consider only these metrics 'wireline gross adds', 'wireless gross adds', 'wireline net adds', 'wireless net adds', 'wireline disconnects', 'wireless disconnects', 'wireline churn rate', 'wireless churn rate'.
    9. what to return or group by. For example, if the user asks for 'top 3 regions', then return 'region', to performing months then return rpt_mth, top performing services then return service_type, top performing products means return product_type. If the user asks for 'best performing product types', then return 'product_type'. If not specified, leave empty.
    10. number of items if user asks for top N or bottom N. For example if the user asks for 'top 3 regions', then return '3', . If not specified, leave empty.
    11. order if user asks for top or bottom. For example if the user asks for 'top 3 regions', then return 'DESC'. If the user asks for 'bottom 3 regions', then return 'ASC'. If not specified, leave empty.

    Respond strictly in JSON format with keys: 'metrics', 'relationship_type','product_type','service_type','regions', 'time_period','year', 'group_by', 'limit', 'order'.
    Example:
    {{
      "metrics": ["customer churn", "ARPU"],
      "relationship_type": ["influenced"],
      "product_type": ["wireline"],
      "service_type": ["voice","data"],
      "regions": ["southeast"],
      "time_period": ["january"],
      "year": [2024],
      "group_by": ["region"],
      "limit": 3,
      "order": "DESC"
    }}
    """

    # Using 'generate_content' for Gemini, and specifying response_mime_type for JSON output
    response = client.generate_content(
        prompt,
        generation_config={
            "response_mime_type": "application/json"
        }
    )
    raw_output = ""
    # Safely access the text content
    if response.candidates and len(response.candidates) > 0 and response.candidates[0].content and len(response.candidates[0].content.parts) > 0:
        raw_output = response.candidates[0].content.parts[0].text
    else:
        finish_reason = response.candidates[0].finish_reason if response.candidates else 'UNKNOWN'
        st.warning(f"Model returned no content for entity extraction. Finish reason: {finish_reason}")
        return {} # Return empty dict if no content

    try:
        parsed_json = json.loads(raw_output)
        print(f"Extracted JSON: {parsed_json}")  # Debugging output
        return parsed_json
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse model output as JSON. Raw output: ```{raw_output}``` Error: {e}")
        raise e
    except Exception as e:
        st.error(f"An unexpected error occurred during JSON parsing: {e}")
        raise e

# --- Function: Generate Cypher Query for Reasoning Questions (existing) ---
def generate_cypher(parsed: dict) -> str:
    try:
        metrics = [m.lower() for m in (parsed.get('metrics') or [])]
        relation = [m.lower() for m in (parsed.get('relationship_type') or [])]
        period = [m.lower() for m in (parsed.get('time_period') or [])]
        year = [m.lower() if isinstance(m, str) else m for m in (parsed.get('year') or [])]
        product_name = [m.lower() for m in (parsed.get('product_type') or [])]
        service_name = [m.lower() for m in (parsed.get('service_type') or [])]
        regions = [r.lower() for r in (parsed.get('regions') or [])]

        # Handle current date for dynamic period/year
        current_date = datetime.now()

        # Adjusted year logic for 'last year'
        if "last year" in [str(y).lower() for y in year]: # Handle case where year might be int or string
            year = [current_date.year - 1]
        elif not year:
            year = [current_date.year]

        # Adjusted period logic for 'last month'
        if "last month" in period:
            last_month_date = current_date.replace(day=1) - timedelta(days=1)
            period = [last_month_date.strftime("%B").lower()] # e.g., "may"
            year = [last_month_date.year] # Ensure year is updated if period is 'last month'
        elif "this year" in period or "rest of the year" in period:
            year = [current_date.year]
            period = ["full year"]
        elif not year and "month" in period: # if only month is given, assume current year
            year = [current_date.year]

        if period==[''] or not period:
            period = ["full year"]

        if service_name==[''] and period==['full year']: # If no service type and full year, assume full year for service type data
            service_name = ["full year"]
        if regions==[''] and period==['full year']: # If no region and full year, assume full year for region data
            regions = ["full year"]

        # Convert all elements in the year list to integers if possible
        if year:
            try:
                year = [int(y) for y in year if str(y).strip() != ""]
            except Exception as e:
                st.warning(f"Could not convert year values to integers: {e}")

        conditions = []
        # Ensure year is treated as an integer for Cypher
        year_condition_value = year if year else None

        if period and period!=['']:
            conditions.append(f"toLower(md1.rpt_mth) in {json.dumps(period)} AND toLower(md2.rpt_mth) in {json.dumps(period)}")
        if year_condition_value:
            conditions.append(f"md1.rpt_year in {json.dumps(year_condition_value)} AND md2.rpt_year in {json.dumps(year_condition_value)}")
        if product_name and product_name!=['']:
            conditions.append(f"toLower(md1.product_name) in {json.dumps(product_name)} AND toLower(md2.product_name) in {json.dumps(product_name)}")

        # Corrected logic for service_name and regions
        if service_name and service_name!=['']:
            conditions.append(f"toLower(md1.service_type) in {json.dumps(service_name)} AND toLower(md2.service_type) in {json.dumps(service_name)}")
        elif service_name==[] and regions ==[]: # If both service and region are empty, default to 'full year' for both
            conditions.append(f"toLower(md1.service_type) in {json.dumps(period)} AND toLower(md2.service_type) in {json.dumps(period)}")
            conditions.append(f"toLower(md1.region) in {json.dumps(period)} AND toLower(md2.region) in {json.dumps(period)}")

        if regions and regions!=['']:
            conditions.append(f"toLower(md1.region) IN {json.dumps(regions)} AND toLower(md2.region) IN {json.dumps(regions)}")
        elif regions==[] and service_name==[]:
            pass # Already added in the combined condition
        elif regions==[]: # If region is empty but service is not, then default region to full year based on period
            conditions.append(f"toLower(md1.region) in {json.dumps(period)} AND toLower(md2.region) in {json.dumps(period)}")



        conditions.append(f"""
            toLower(md1.service_type) = toLower(md2.service_type)
            and toLower(md1.product_name) = toLower(md2.product_name)
            and toLower(md1.region) = toLower(md2.region)""")

        # Base query for metrics and their relationships
        query = f"""
        MATCH (m1:Metric)-[r*1]->(m2:Metric)
        WHERE toLower(m1.name) IN {json.dumps(metrics)} OR toLower(m2.name) IN {json.dumps(metrics)}
        WITH m1, m2, r
        MATCH (m1)-[has_data_m1_md1:HAS_DATA]->(md1:MetricData)
        MATCH (m2)-[has_data_m2_md2:HAS_DATA]->(md2:MetricData)
        """

        if conditions:
            # Filter out any empty conditions that might have snuck in due to logic branches
            clean_conditions = [c for c in conditions if c.strip()]
            if clean_conditions:
                query += f"WHERE " + " AND ".join(clean_conditions) + "\n"

        query += f"""
        RETURN DISTINCT m1.name AS Metric_A, toFloat(md1.metric_value) AS Metric_A_Value, md1.rpt_mth AS ReportMonth, md1.rpt_year AS ReportYear,
                        m2.name AS Metric_B, toFloat(md2.metric_value) AS Metric_B_Value,
                        md1.service_type , md1.product_name ,md1.region ,
                        [rel IN r | rel.reasoning] as reasoning,md1.summary,md2.summary//,r,m1,m2,md1,md2,has_data_m1_md1,has_data_m2_md2
        """
        return query
    except Exception as e:
        st.error(f"Failed to generate cypher query: {e}")
        traceback.print_exc()
        print(f"Error executing Neo4j query: {e}")
        return "" # Return empty string on error

# --- Function: Generate Cypher Query for Simple Questions (NEW) ---
def generate_simple_cypher(parsed: dict) -> str:
    """
    Generates a Cypher query to fetch a specific metric's value based on provided dimensions,
    and also supports aggregation (e.g., top N regions).
    """
    metrics = [m.lower() for m in (parsed.get('metrics') or [])]
    period = [m.lower() for m in (parsed.get('time_period') or [])]
    year = [m.lower() if isinstance(m, str) else m for m in (parsed.get('year') or [])]
    product_name = [m.lower() for m in (parsed.get('product_type') or [])]
    service_name = [m.lower() for m in (parsed.get('service_type') or [])]
    regions = [r.lower() for r in (parsed.get('regions') or [])]
    group_by_dimension = parsed.get('group_by', [])
    limit_results = parsed.get('limit')
    order_by_direction = parsed.get('order') # "ASC" or "DESC"

    if not metrics:
        return "" # Cannot generate query without a metric

    metric_name = metrics[0] # Assume the first metric is the target

    conditions = []
    # Handle current date for dynamic period/year
    current_date = datetime.now()

    if "last year" in [str(y).lower() for y in year]:
        year = [current_date.year - 1]
    elif not year:
        year = [current_date.year]

    if "last month" in period:
        last_month_date = current_date.replace(day=1) - timedelta(days=1)
        period = [last_month_date.strftime("%B").lower()]
        year = [last_month_date.year]
    elif "this year" in period:
        year = [current_date.year]
        period = ["full year"]
    elif not year and "month" in period:
        year = [current_date.year]

    if period == [''] or not period:
        period = ["full year"]
    
            # Convert all elements in the year list to integers if possible
    if year:
        try:
            year = [int(y) for y in year if str(y).strip() != ""]
            print(year)
        except Exception as e:
            st.warning(f"Could not convert year values to integers: {e}")
    else:
        print(year)

    if period and period != ['']:
        conditions.append(f"toLower(md.rpt_mth) in {json.dumps(period)}")
    if year:
        conditions.append(f"md.rpt_year in {json.dumps(year)}")
    if product_name and product_name != ['']:
        conditions.append(f"toLower(md.product_name) in {json.dumps(product_name)}")

    if service_name and service_name != ['']:
        conditions.append(f"toLower(md.service_type) in {json.dumps(service_name)}")

    # Only add region filter if specific regions are provided and we are NOT grouping by region
    if regions ==["all regions"] and group_by_dimension == ["region"]:
        conditions.append(f' NOT toLower(md.region) IN ["all regions"]') # This is to handle the case where user wants all regions
    elif regions and regions != [] and group_by_dimension != ["region"]:
        conditions.append(f"toLower(md.region) in {json.dumps(regions)}")

    query_parts = [
        "MATCH (m:Metric)-[:HAS_DATA]->(md:MetricData)",
        f"WHERE toLower(m.name) = '{metric_name.lower()}'"
    ]

    if conditions:
        query_parts.append("AND " + " AND ".join(conditions))
    print("group by ",group_by_dimension)
    # Handle grouping, ordering, and limiting for "top N" or "bottom N" queries
    if group_by_dimension:
        dimension_to_group_by = group_by_dimension[0] if group_by_dimension else None

        if dimension_to_group_by:
            # The alias for the grouping dimension in WITH and RETURN clauses
            grouping_alias = f"md.{dimension_to_group_by}"
            
            query_parts.append(f"WITH m, md, {grouping_alias} AS GroupingKey")
            
            # The SUM alias for ordering, converting metric_value to float
            sum_alias = f"{metric_name.replace(' ', '_').title()}TotalValue"
            
            return_items = [
                f"GroupingKey AS GroupingDimension",
                f"SUM(toFloat(md.metric_value)) AS {sum_alias}" # Convert to float for SUM
            ]
            
            # Add other return dimensions only if they are not the grouping dimension
            # and if they make sense to COLLECT.
            if dimension_to_group_by != 'product_name':
                return_items.append("COLLECT(DISTINCT md.product_name) AS ProductTypes")
            if dimension_to_group_by != 'service_type':
                return_items.append("COLLECT(DISTINCT md.service_type) AS ServiceTypes")
            if dimension_to_group_by != 'region':
                return_items.append("COLLECT(DISTINCT md.region) AS Regions")
            if dimension_to_group_by != 'rpt_mth':
                return_items.append("COLLECT(DISTINCT md.rpt_mth) AS ReportMonths")
            if dimension_to_group_by != 'rpt_year':
                return_items.append("COLLECT(DISTINCT md.rpt_year) AS ReportYears")
            
            query_parts.append(f"RETURN {', '.join(return_items)}")
            query_parts.append(f"ORDER BY {sum_alias} {order_by_direction if order_by_direction else 'DESC'}")
            if limit_results:
                query_parts.append(f"LIMIT {limit_results}")
        else: # Should not happen if group_by_dimension is not empty, but for safety
            query_parts.append(f"""
            RETURN md.summary,m.name AS MetricName, toFloat(md.metric_value) AS MetricValue, md.rpt_mth AS ReportMonth,
                   md.rpt_year AS ReportYear, md.product_name AS ProductType,
                   md.service_type AS ServiceType, md.region AS Region
            """)
    else: # Standard simple query without aggregation
        query_parts.append(f"""
        RETURN md.summary //m.name AS MetricName, toFloat(md.metric_value) AS MetricValue, md.rpt_mth AS ReportMonth,
               //md.rpt_year AS ReportYear, md.product_name AS ProductType,
               //md.service_type AS ServiceType, md.region AS Region
        """)
        
    return "\n".join(query_parts)


# --- Function: Run Neo4j Query ---
def query_neo4j(cypher_query: str):
    """
    Executes a Cypher query and captures Neo4j graph traversal metrics such as number of nodes traversed,
    db hits, rows, and time taken (where available).
    Returns: (data, metrics_dict)
    """
    if not driver:
        st.error("Neo4j driver not initialized. Cannot query database.")
        return [], {}
    try:
        with driver.session() as session:
            # --- Execute the Actual Query First ---
            result = session.run(cypher_query)
            data = [record.data() for record in result]

            # --- Get Query Plan and Profile Metrics ---
            plan_metrics = {}
            try:
                # Use PROFILE to get actual execution metrics (if supported)
                profile_result = session.run("PROFILE " + cypher_query)
                profile_summary = profile_result.consume()
                plan = profile_summary.profile

                def extract_metrics(plan_node):
                    # Recursively extract metrics from the plan tree
                    args = plan_node.get("args", {})
                    children = args.get("children", [])
                    # Convert all metric values to int if possible, else None
                    def safe_int(val):
                        try:
                            return int(val)
                        except Exception:
                            return 0
                    db_hits = safe_int(args.get("DbHits"))
                    rows = safe_int(args.get("Rows"))
                    page_cache_hits = safe_int(args.get("PageCacheHits"))
                    page_cache_misses = safe_int(args.get("PageCacheMisses"))
                    identifiers = getattr(plan_node, "identifiers", [])
                    children_metrics = []
                    children_db_hits_sum = 0
                    for c in children:
                        child_metrics = extract_metrics(c)
                        children_metrics.append(child_metrics)
                        children_db_hits_sum += child_metrics.get("db_hits_total", 0)
                    db_hits_total = db_hits + children_db_hits_sum
                    metrics = {
                        "rows": rows,
                        "page_cache_hits": page_cache_hits,
                        "page_cache_misses": page_cache_misses,
                        "db_hits_total": db_hits_total
                    }
                    return metrics

                if plan:
                    plan_metrics = extract_metrics(plan)
                else:
                    plan_metrics = {"info": "No profile plan returned."}

                # Get summary-level metrics
                summary_metrics = {
                    "result_available_after_ms": getattr(profile_summary, "result_available_after", None),
                    "result_consumed_after_ms": getattr(profile_summary, "result_consumed_after", None),
                }
                plan_metrics.update(summary_metrics)

            except Exception as profile_e:
                plan_metrics = {"error": f"Failed to get PROFILE metrics: {profile_e}"}
                print(f"PROFILE error: {profile_e}")

            return data, plan_metrics
    except Exception as e:
        st.error(f"Failed to execute Neo4j query: {e}")
        traceback.print_exc()
        print(f"Error executing Neo4j query: {e}")
        return [], {}


#def generate_student_simple_answer_ollama(user_question: str, data: list, ollama_model_name: str = "gemma2:2b") -> str:
def format_data_for_llm(data: list) -> str:
    """
    Converts a list of dictionaries into a simple, human-readable string
    that is easier for an LLM to parse than raw JSON.

    Example Output:
    - region: Mumbai, wireline_gross_adds: 4500
    - region: Delhi, wireline_gross_adds: 4250
    """
    if not data:
        return "No data provided."
    
    # Creates a simple "key: value" string for each dictionary item
    formatted_lines = []
    for item in data:
        line = ", ".join([f"{key}: {value}" for key, value in item.items()])
        formatted_lines.append(f"- {line}")
        
    return "\n".join(formatted_lines)


def build_simple_answer(data: list, user_question: str) -> str:
    """
    Generates a concise answer for simple metric lookup questions using Ollama,
    optimized for a Gemma-based model with a robust prompt structure.

    Args:
        user_question (str): The user's question.
        data (list): The contextual data to base the answer on.
    Returns:
        str: The generated answer from the LLM.
    """
    print("\n--- Starting build_simple_answer function ---")
    ollama_model_name = STUDENT_MODEL 

    try:
        ollama.list()
        print(f"Ollama server is reachable. Using model: {ollama_model_name}")
    except Exception as e:
        print(f"FATAL: Error connecting to Ollama server: {e}")
        return "Ollama server is not running or the model is unavailable."

    if not data:
        print("Warning: Context data is empty.")
        return "No data found for your specific query."

    # --- FIX 1: Format data into a simple, clean text format instead of JSON ---
    formatted_data = format_data_for_llm(data)
    print("\n--- Formatted Context Data for LLM ---")
    print(formatted_data)
    print("--------------------------------------\n")
    
    # --- FIX 2: Refine the system prompt for absolute clarity ---
    # All instructions, including formatting, are here.
    system_instructions = """You are an expert assistant that answers questions using ONLY the provided context. Follow all rules strictly.

### Core Rules
1.  **Strictly Context-Based:** Base every part of your answer directly on the `CONTEXT` provided.
2.  **Handle Missing Information:** If the answer is not in the context, respond ONLY with: `The answer is not available in the context.`
3.  **Be Direct:** Do not use conversational filler like "Here is the answer". Go straight to the fact.
4.  **Default Metrics:** For general "performance" questions, ONLY use: 'wireline gross adds', 'wireless gross adds', 'wireline net adds', 'wireless net adds', 'wireline disconnects', 'wireless disconnects', 'wireline churn rate', 'wireless churn rate'.
5.  **No Summing:** Use totals provided in the context. Do not calculate sums yourself.
6.  **Include Values:** Always state the metric's value from the context.
7. **No Hallucination:** Do not invent or assume values. Use only what is in the context.
8. **if question asked for top 3 or top 5 or 5 bullet points and if we do not have data for that then show only what we have.
### Formatting Rules
- **Top N Lists:** For questions asking for a "top N", use this exact format:
The top [N] regions with the most [Metric Name] are:
* [Region 1 Name]: [Value]
* [Region 2 Name]: [Value]
* [Region 3 Name]: [Value]

- **Single Number Answers:** State the fact clearly. Example: `Wireline gross adds in January were 9121.0.`"""

    # --- FIX 3: Drastically simplify the user message structure ---
    # The user prompt should ONLY contain the data and the question, clearly separated.
    user_content = f"""CONTEXT:
{formatted_data}

QUESTION:
{user_question}"""

    messages = [
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": user_content},
    ]

    print("--- Sending Request to Ollama ---")
    print(f"System Prompt Length: {len(system_instructions)}")
    print(f"User Content Length: {len(user_content)}")
    # To debug, you can print the full messages list: print(json.dumps(messages, indent=2))
    print("-----------------------------------")
    
    try:
        response = ollama.chat(
            model=ollama_model_name,
            messages=messages,
            options={
                "temperature": 0.1,
                "top_k": 20, # Can be lowered for more deterministic tasks
                "top_p": 0.9,
                "repeat_penalty": 1.15, # Slightly increased penalty
                "num_predict": 500,
                "stop": ["<end_of_turn>", "<start_of_turn>user", "QUESTION:", "CONTEXT:"]
            }
        )
        
        print("\n--- Raw Response from Ollama ---")
        print(response)
        print("--------------------------------\n")

        answer = response['message']['content'].strip()
        
        if not answer:
            print("Warning: Ollama generated an empty answer.")
            return "Ollama generated an empty answer. The model may need further tuning."

        print(f"Final Generated Answer: {answer}")
        return answer

    except Exception as e:
        print(f"Error during Ollama generation: {e}")
        traceback.print_exc()
        return f"An error occurred while generating the answer with Ollama: {e}"
        
def build_narrative(data: list, user_question: str) -> str:

    """
    Generates a concise answer for simple metric lookup questions using Ollama,
    formatted for Gemma 2, with improved repetition prevention.

    Args:
        user_question (str): The user's question.
        data (list): The contextual data to base the answer on.
    Returns:
        str: The generated answer from the LLM.
    """
    print("\n--- Starting generate_student_simple_answer_ollama function ---")

    try:
        # Basic check for Ollama connectivity
        # This will raise an exception if the server isn't running or the model isn't available
        # You can also use ollama.list() to verify model existence if needed.
        ollama_model_name = STUDENT_MODEL
        
        ollama.list()
        print(f"Ollama server is reachable. Attempting to use model: {ollama_model_name}")
    except Exception as e:
        print(f"Error connecting to Ollama server or listing models: {e}")
        return "Ollama server might not be running or the specified model is not found. Please check your Ollama setup."

    if not data:
        print("Warning: No data found for your specific query.")
        return "No data found for your specific query. Please check the metric, dimensions, and time period."

    # Prepare a compact JSON of the data for the LLM
    compact_data = json.dumps(data, indent=2)
    print(f"Compact Data (first 200 chars): {compact_data[:200]}...")

    # Define the system instructions.
    system_instructions = """        Using the following data and original question, write an insight story with the metric and metric value.
        Focus on causality, comparison, and trends. If the answer generated using multiple metrics then write up to 10 lines other wise Write 4-5 lines only.
        If the user query is asking for general performance or scorecard and did not mention any metric name then consider only these metrics 'wireline gross adds', 'wireless gross adds', 'wireline net adds', 'wireless net adds', 'wireline disconnects', 'wireless disconnects', 'wireline churn rate', 'wireless churn rate'.
        If the user query asks to write in bullet points then write in bullet points.
        Do not sum up group(same metric with various dimensions) Metric values if direct metric avaialable with value.
        Incase we need to sum some of the metrics then use proper justification to sum up the values accurately.
        Include metric and metric value while calclualting the total value in the narrative.
        Generate a narrative with avaialabel context only , no hallucinated values.
        When no specific product type mentioned in user query , then concider all products for narration.
        If service type mentioned then mention figures for each service type, otherwise mention figures for all services.
        you must include  metric values in the narrative.

"""
    # Combine context and question into the user message
    full_user_content = f"Here is the context:\n\n{compact_data}\n\n### Question:\n\n{user_question}\n\n### Answer:"
    print(f"Full User Content length: {len(full_user_content)} characters.")

    # Create the messages list for Ollama's chat endpoint
    messages = [
    {"role": "system", "content": system_instructions},
    {"role": "user", "content": f"{full_user_content}"}, # The model_input_text is your combined context and question
    ]
    print("Messages list prepared for Ollama.")

    try:
        print("Sending request to Ollama chat endpoint...")
        # Call Ollama's chat endpoint
        response = ollama.chat(
            model=ollama_model_name,
            messages=messages,
            options={
                # Core parameters to combat repetition:
                "temperature": 0.7,      # Increase temperature slightly for more variability (0.0 means no randomness, highly prone to repetition)
                "top_k": 40,             # Consider top_k most probable tokens
                "top_p": 0.9,            # Consider tokens that sum up to top_p probability
                "repeat_penalty": 1.1,   # Penalize repeating tokens more (e.g., 1.1 or 1.2)
                "num_predict": 500,      # Max new tokens (similar to max_new_tokens)
            }
        )
        print("Response received from Ollama.")
        print("Response form llm",response)
        # Extract the content from the response
        answer = response['message']['content'].strip()

        # Gemma models sometimes add an extra "Answer:" at the start when instructions are too strict.
        # Let's remove it if it appears.
        if answer.startswith("Answer:"):
            answer = answer[len("Answer:"):].strip()

        if not answer:
            print("Warning: Ollama generated an empty answer.")
            return "Ollama generated an empty answer. Try adjusting the prompt or model."

        print(f"Final Generated Answer (Ollama): {answer}")
        return answer

    except Exception as e:
        print(f"Error during Ollama generation: {e}")
        traceback.print_exc()
        return f"An error occurred while generating the answer with Ollama: {e}"



# --- Function: Validate Answer with AI (Gemini 2.5 Flash Safe) ---
def validate_answer_with_ai(user_question: str, generated_narrative: str, raw_data: list) -> str:
    """
    Validates the generated narrative using simplified, Gemini 2.5 compatible prompt.
    Uses a summarized version of raw_data to prevent response block or truncation.
    """
    if not client:
        return "LLM client not initialized. Cannot validate answer."
    try:
        # Summarize raw data (limit to first 3 records, or fewer for very large data)
        summary_data = raw_data # Consider raw_data[:5] or a more sophisticated summary if data is huge
        if not summary_data:
            return "Validation: ERROR - No data available for validation."

        compact_data = json.dumps(summary_data, indent=2)

        # Gemini-safe prompt
        validation_prompt = f"""
You are an assistant validating if the narrative is supported by data.

User Question:
{user_question}

Narrative:
{generated_narrative}

Key Data:
{compact_data}

Does the narrative align with this data? if any metric summed up with its dimensions as long as the sum is matching with its dimensions accept it.
Reply with "Validation: SUCCESS" if it matches the data well,
or "Validation: FAILURE - [Brief reason]" if something important is missing or incorrect.
        """.strip()

        validation_response = client.generate_content(
            validation_prompt
        )
        validation_result = ""
        if validation_response.candidates and len(validation_response.candidates) > 0:
            candidate = validation_response.candidates[0]
            if candidate.content and len(candidate.content.parts) > 0:
                validation_result = candidate.content.parts[0].text.strip()
            else:
                finish_reason = candidate.finish_reason or "UNKNOWN"
                st.warning(f"Validation response returned empty content. Finish reason: {finish_reason}")
                return f"Validation: ERROR - Empty model response (Finish reason: {finish_reason})"
        else:
            st.warning("Validation model returned no candidates.")
            return "Validation: ERROR - No candidates returned."

        return validation_result

    except Exception as e:
        st.error(f"Failed to validate answer: {e}")
        traceback.print_exc()
        return "Validation: ERROR - Exception occurred during validation."

# --- NEW Function: Generate Ground Truth from Data ---
def generate_ground_truth_from_data(user_question: str, raw_data: list) -> str:
    """
    Generates a concise, factual ground truth answer based *only* on the provided raw data.
    This function leverages an LLM (similar to your validation function) but with a different prompt.
    """
    if not client:
        return "LLM client not initialized. Cannot generate ground truth."
    if not raw_data:
        return "No relevant data found for this query to generate a ground truth."

    # Use a compact representation of raw_data to stay within token limits
    # Limiting to 5 records as a heuristic, adjust based on typical data size and model context window
    #compact_data = json.dumps(raw_data[:5], indent=2)
    compact_data = json.dumps(raw_data, indent=2)

    # The key is this prompt: it instructs the LLM to act as a 'ground truth' generator.
    ground_truth_prompt = f"""
        Focus on causality, comparison, and trends. If the answer generated using multiple metrics then write up to 10 lines other wise Write 4-5 lines only.
        If the user query is asking for general performance or scorecard and did not mention any metric name then consider only these metrics 'wireline gross adds', 'wireless gross adds', 'wireline net adds', 'wireless net adds', 'wireline disconnects', 'wireless disconnects', 'wireline churn rate', 'wireless churn rate'.
        If the user query asks to write in bullet points then write in bullet points.
        Do not sum up group(same metric with various dimensions) Metric values if direct metric avaialable with value.
        Incase we need to sum some of the metrics then use proper justification to sum up the values accurately.
        Include metric and metric value while calclualting the total value in the narrative.
        Generate a narrative with avaialabel context only , no hallucinated values.
        When no specific product type mentioned in user query , then concider all products for narration.
        If service type mentioned then mention figures for each service type, otherwise mention figures for all services.
        If the question cannot be answered from the provided 'Raw Data', state that clearly.

    User Question:
    {user_question}

    Raw Data:
    {compact_data}

    Ground Truth Answer:
    """
    try:
        gt_response = client.generate_content(
            ground_truth_prompt
        )
        if gt_response.candidates and len(gt_response.candidates) > 0 and gt_response.candidates[0].content and len(gt_response.candidates[0].content.parts) > 0:
            print(f"Ground truth response: {gt_response.candidates[0].content.parts[0].text.strip()}" )
            return gt_response.candidates[0].content.parts[0].text.strip()
        else:
            # Handle cases where LLM returns no content for ground truth
            finish_reason = gt_response.candidates[0].finish_reason if gt_response.candidates else 'UNKNOWN'
            st.warning(f"Ground truth generation model returned no content. Finish reason: {finish_reason}. Defaulting to empty string.")
            return "" # Default to empty string if no ground truth generated
    except Exception as e:
        print(f"Error generating ground truth: {e}")
        traceback.print_exc()
        return f"Error generating ground truth: {e}"


# --- LangChain Tool for Query Relevance Validation ---
def _check_telecom_relevance(query: str) -> str:
    """
    Internal helper function for the LangChain tool.
    Determines if a query is related to telecom business metrics.
    Returns "RELEVANT" or "IRRELEVANT".
    """
    if not client:
        return "LLM client not initialized. Cannot check relevance."
    try:
        # Simplified prompt with few-shot example for clear output format
        relevance_prompt = f"""
        Is the following user query related to telecom business metrics or KPIs?
        Allow queries related to metadata or help related such as hi or help , which data you have etc
        if user query hasmultiple questions , all should be businessrelated
        Query: "{query}"

        Examples:
        Query: "What is the customer churn rate?"
        Response: RELEVANT

        Query: "Tell me a joke."
        Response: IRRELEVANT

        Query: "How is ARPU performing this quarter?"
        Response: RELEVANT

        Query: "How we are doing in q2?"
        Response: RELEVANT

        Query: "Performance summary or score card for this year?"
        Response: RELEVANT

        Query: "What products you can support?"
        Response: RELEVANT

        Query: "What dimensions you have"
        Response: RELEVANT

        Query: "Hi"
        Response: RELEVANT

        Query: "Help"
        Response: RELEVANT

        Response:
        """
        relevance_response = client.generate_content(
            relevance_prompt
        )
        relevance_result = ""
        if relevance_response.candidates and len(relevance_response.candidates) > 0 and relevance_response.candidates[0].content and len(relevance_response.candidates[0].content.parts) > 0:
            relevance_result = relevance_response.candidates[0].content.parts[0].text.strip().upper()
        else:
            finish_reason = relevance_response.candidates[0].finish_reason if relevance_response.candidates else 'UNKNOWN'

            # Check for safety feedback first
            if relevance_response.prompt_feedback and relevance_response.prompt_feedback.safety_ratings:
                safety_reasons = ", ".join([
                    f"{rating.category.name}: {rating.probability.name}"
                    for rating in relevance_response.prompt_feedback.safety_ratings
                    if rating.blocked
                ])
                st.warning(f"Query relevance check blocked due to safety policies. Reasons: {safety_reasons}. Defaulting to RELEVANT to allow processing.")
                return "RELEVANT" # Default to relevant if blocked by safety

            # If not safety, but still no content (e.g., STOPPED for other reasons)
            st.warning(f"Model returned no content for query relevance check. Finish reason: {finish_reason}. Defaulting to RELEVANT.")
            return "RELEVANT"

        return relevance_result # Return the exact string "RELEVANT" or "IRRELEVANT"
    except Exception as e:
        st.error(f"Failed to validate query relevance: {e}")
        traceback.print_exc()
        print(f"Error validating query relevance: {e}")
        # Default to relevant to allow processing if validation itself fails
        return "RELEVANT"

# --- NEW Function: classify_question_type ---
def classify_question_type(query: str) -> str:
    """
    Classifies a user query into specific types to determine the appropriate processing flow.
    Returns:
        - 'SIMPLE' for direct metric lookups (e.g., "what is ARPU for May")
        - 'REASONING' for questions requiring relationship analysis (e.g., "how does churn affect revenue")
        - 'METADATA_SERVICE_TYPE:PRODUCT_NAME' if asking for service types for a specific product (e.g., "what service types for wireline")
        - 'METADATA_AVAILABILITY' for general data availability questions (e.g., "what data do you have", "what metrics are available", "what regions data we have", "do we have data for X month", "do you have data for Y product", "hi", "help")
    """
    if not client:
        return "LLM client not initialized. Cannot classify question type."
    try:
        # Prompt for classifying the question type
        classification_prompt = f"""
        Classify the following user query into one of these categories:
        - 'SIMPLE': If it's a direct question asking for a specific metric's value, possibly with dimensions (e.g., "What is the churn rate for May?", "ARPU for Q1 2024 in Texas", "Top 3 regions by wireless gross adds").
        - 'REASONING': If it's a question asking about relationships, causality, or trends between metrics (e.g., "How does churn affect revenue?", "What is the relationship between ARPU and customer satisfaction?").
        - 'METADATA_SERVICE_TYPE:PRODUCT_NAME': If the user explicitly asks for 'service types' for a *specific product* (e.g., "what service types for wireline?", "what services does wireless have?"). The response should be in the format 'METADATA_SERVICE_TYPE:product_name_in_lowercase'.
        - 'METADATA_AVAILABILITY': If the user asks general questions about data availability, help, or lists of available metrics, products, regions, time periods, or dimensions (e.g., "what data do you have?", "what metrics are available?", "what dimensions can I query by?", "hi", "help", "what regions data we have", "do we have data for X month", "do you have data for Y product", "performance summary" or "scorecard").
        - 'UNKNOWN': If it doesn't fit any of the above.

        User Query: "{query}"

        Examples:
        User Query: "What metrics do you have?"
        Classification: METADATA_AVAILABILITY

        User Query: "Hi"
        Classification: METADATA_AVAILABILITY

        User Query: "Help"
        Classification: METADATA_AVAILABILITY

        User Query: "What is the customer churn rate in June?"
        Classification: SIMPLE

        User Query: "How does wireline gross adds affect wireline revenue?"
        Classification: REASONING

        User Query: "What service types for wireline?"
        Classification: METADATA_SERVICE_TYPE:wireline

        User Query: "Show me products."
        Classification: METADATA_AVAILABILITY

        User Query: "Can you give me a performance summary for this year?"
        Classification: METADATA_AVAILABILITY

        User Query: "What are the top 5 regions for wireless net adds?"
        Classification: SIMPLE

        Classification:
        """
        classification_response = client.generate_content(
            classification_prompt
        )
        classification_result = ""
        if classification_response.candidates and len(classification_response.candidates) > 0 and classification_response.candidates[0].content and len(classification_response.candidates[0].content.parts) > 0:
            classification_result = classification_response.candidates[0].content.parts[0].text.strip().upper()
        else:
            finish_reason = classification_response.candidates[0].finish_reason if classification_response.candidates else 'UNKNOWN'
            st.warning(f"Model returned no content for question classification. Finish reason: {finish_reason}. Defaulting to UNKNOWN.")
            return "UNKNOWN" # Default to UNKNOWN if no content

        # Additional parsing for METADATA_SERVICE_TYPE to ensure product name is lowercase
        if classification_result.startswith("METADATA_SERVICE_TYPE:"):
            parts = classification_result.split(":")
            if len(parts) == 2:
                product_name_raw = parts[1].strip()
                # Basic sanitization to get a usable product name
                product_name_clean = re.sub(r'[^a-zA-Z0-9\s]', '', product_name_raw).strip().lower()
                return f"METADATA_SERVICE_TYPE:{product_name_clean}"
            else:
                st.warning(f"Unexpected format for METADATA_SERVICE_TYPE classification: {classification_result}. Defaulting to METADATA_AVAILABILITY.")
                return "METADATA_AVAILABILITY" # Fallback if product name is malformed

        return classification_result

    except Exception as e:
        st.error(f"Failed to classify question type: {e}")
        traceback.print_exc()
        print(f"Error classifying question type: {e}")
        return "UNKNOWN" # Return UNKNOWN on error

# --- New Function: Get Available Metadata ---
def get_available_metadata_from_neo4j(metadata_type: str = "all", product_name: str = None) -> tuple[str, str, dict, list]: # Added 'list' to return type
    """
    Fetches available metrics, product types, service types, regions, and report months/years from Neo4j.
    Converts the fetched data to a more readable format (e.g., title case) before returning.
    The 'metadata_type' parameter allows fetching specific categories of metadata.
    Valid values for metadata_type: "all", "metrics", "dimensions", "product_types", "service_types", "regions", "time_periods".
    The 'product_name' parameter allows filtering service types by a specific product.
    Returns a tuple: (formatted_response_string, cypher_query_used, metrics_dict, raw_data_list)
    """
    if not driver:
        st.error("Neo4j driver not initialized. Cannot fetch metadata.")
        return ("I'm sorry, I couldn't retrieve the available data and dimensions at this time due to a database connection issue.", "", {}, [])
    
    response_parts = []
    cypher_queries_executed = []
    metadata_metrics = {}
    raw_metadata_results = [] # New list to store raw data for contexts

    try:
        with driver.session() as session:
            if metadata_type == "all" or metadata_type == "metrics":
                metrics_query = "MATCH (m:Metric) RETURN DISTINCT m.name AS metricName ORDER BY metricName"
                cypher_queries_executed.append(metrics_query)
                metrics_result = session.run(metrics_query)
                metrics_data = [record.data() for record in metrics_result] # Get raw data
                raw_metadata_results.extend(metrics_data) # Add to contexts
                metrics = [record["metricName"].replace('_', ' ').title() for record in metrics_data]
                response_parts.append(f"**Metrics:** {', '.join(metrics) if metrics else 'N/A'}")
                metadata_metrics["metrics_count"] = len(metrics)

            if metadata_type == "all" or metadata_type == "dimensions" or metadata_type == "product_types":
                products_query = "MATCH (md:MetricData) RETURN DISTINCT md.product_name AS productName ORDER BY productName"
                cypher_queries_executed.append(products_query)
                products_result = session.run(products_query)
                products_data = [record.data() for record in products_result] # Get raw data
                raw_metadata_results.extend(products_data) # Add to contexts
                products = [record["productName"].replace('_', ' ').title() for record in products_data]
                response_parts.append(f"**Product Types:** {', '.join(products) if products else 'N/A'}")
                metadata_metrics["product_types_count"] = len(products)

            if metadata_type == "all" or metadata_type == "dimensions" or metadata_type == "service_types":
                service_query_conditions = []
                if product_name:
                    service_query_conditions.append(f"toLower(md.product_name) = '{product_name.lower()}'")
                
                service_query = "MATCH (md:MetricData) "
                if service_query_conditions:
                    service_query += "WHERE " + " AND ".join(service_query_conditions) + " "
                service_query += "RETURN DISTINCT md.service_type AS serviceType ORDER BY serviceType"
                
                cypher_queries_executed.append(service_query)
                services_result = session.run(service_query)
                services_data = [record.data() for record in services_result] # Get raw data
                raw_metadata_results.extend(services_data) # Add to contexts
                services = [record["serviceType"].replace('_', ' ').title() for record in services_data]
                
                if product_name and services:
                    response_parts.append(f"**Service Types for {product_name.title()}:** {', '.join(services)}")
                elif services:
                    response_parts.append(f"**Service Types (All Products):** {', '.join(services)}")
                else:
                    response_parts.append(f"**Service Types:** N/A")
                metadata_metrics["service_types_count"] = len(services)


            if metadata_type == "all" or metadata_type == "dimensions" or metadata_type == "regions":
                regions_query = "MATCH (md:MetricData) RETURN DISTINCT md.region AS regionName ORDER BY regionName"
                cypher_queries_executed.append(regions_query)
                regions_result = session.run(regions_query)
                regions_data = [record.data() for record in regions_result] # Get raw data
                raw_metadata_results.extend(regions_data) # Add to contexts
                regions = [record["regionName"].replace('_', ' ').title() for record in regions_data]
                response_parts.append(f"**Regions:** {', '.join(regions) if regions else 'N/A'}")
                metadata_metrics["regions_count"] = len(regions)

            if metadata_type == "all" or metadata_type == "time_periods":
                time_query = "MATCH (md:MetricData) RETURN DISTINCT md.rpt_mth AS month, md.rpt_year AS year ORDER BY year DESC, month"
                cypher_queries_executed.append(time_query)
                time_result = session.run(time_query)
                time_data = [record.data() for record in time_result] # Get raw data
                raw_metadata_results.extend(time_data) # Add to contexts
                time_periods = []
                for record in time_data:
                    time_periods.append(f"{record['month'].capitalize()} {record['year']}")
                response_parts.append(f"**Available Report Months/Years:** {', '.join(sorted(list(set(time_periods)))) if time_periods else 'N/A'}")
                metadata_metrics["time_periods_count"] = len(set(time_periods))

        final_response_string = ""
        if not response_parts:
            final_response_string = f"I found no data for the requested metadata type: {metadata_type}."
        elif metadata_type == "all":
            final_response_string = "Hello, I am your virtual Agent, I can answer metrics reasoning questions on the following data:\n\n" + "\n\n".join(response_parts)
        elif metadata_type == "dimensions":
            # For 'dimensions', we explicitly omit the "Metrics" line if it was included in the common logic
            dimension_parts = [part for part in response_parts if not part.startswith("**Metrics:**")]
            final_response_string = "**Supported Dimensions:**\n\n" + "\n\n".join(dimension_parts)
        else:
            final_response_string = "\n\n".join(response_parts)

        # Convert raw_metadata_results to a list of more human-readable strings for Ragas context format
        ragas_contexts = []
        for item in raw_metadata_results:
            readable_parts = []
            for k, v in item.items():
                val_str = str(v) if v is not None else "N/A"
                readable_parts.append(f"{k.replace('_', ' ').title()}: {val_str}")
            ragas_contexts.append("; ".join(readable_parts))

        return final_response_string, "\n\n".join(cypher_queries_executed), metadata_metrics, ragas_contexts # Return raw_metadata_results

    except Exception as e:
        st.error(f"Failed to fetch available metadata from Neo4j: {e}")
        traceback.print_exc()
        return ("I'm sorry, I couldn't retrieve the available data and dimensions at this time.", "", {}, [])

# --- Helper function for robust parsing of Action Input ---
def _parse_and_call_metadata_checker(metadata_params_raw: Union[str, dict]):
    """
    Parses the raw input for AvailableMetadataChecker to ensure it's a dictionary.
    Handles cases where the LLM might output a string representation of a dictionary.
    """
    if isinstance(metadata_params_raw, dict):
        # Already a dictionary, no parsing needed
        parsed_params = metadata_params_raw
    elif isinstance(metadata_params_raw, str):
        try:
            # Try to parse as JSON first
            parsed_params = json.loads(metadata_params_raw)
        except json.JSONDecodeError:
            try:
                # If not JSON, try to parse as a Python literal
                parsed_params = ast.literal_eval(metadata_params_raw)
            except (SyntaxError, ValueError) as e:
                # If neither works, log error and return default/handle gracefully
                st.error(f"Failed to parse Action Input for AvailableMetadataChecker: {metadata_params_raw}. Error: {e}")
                # Fallback to an empty dictionary, which get_available_metadata_from_neo4j can process with defaults
                parsed_params = {}
    else:
        # Unexpected type, treat as empty
        st.warning(f"Unexpected type for metadata_params_raw: {type(metadata_params_raw)}. Expected string or dict.")
        parsed_params = {}

    # Ensure parsed_params is a dictionary before proceeding
    if not isinstance(parsed_params, dict):
        st.error(f"Parsed metadata_params is not a dictionary: {parsed_params}. Defaulting to empty dict.")
        parsed_params = {}

    # Unpack the tuple to get all returned values, including raw_metadata_results
    formatted_response, cypher_query, metrics, raw_results = get_available_metadata_from_neo4j(
        metadata_type=parsed_params.get("metadata_type", "all"),
        product_name=parsed_params.get("product_name")
    )
    # This tool function itself returns a string. The raw_results will be passed through
    # to the main execution flow to be used for Ragas contexts.
    return formatted_response


# Define the LangChain Tools
# Ensure agent_llm is initialized before creating tools that use it.
if agent_llm: # This check relies on the config_ok block above
    tools = [
        Tool(
            name="TelecomQueryRelevanceChecker",
            func=_check_telecom_relevance,
            description="Checks if a user query is related to telecom business metrics or KPIs. Input should be the user's query as a string. Returns 'RELEVANT' or 'IRRELEVANT'."
        ),
        Tool(
            name="AvailableMetadataChecker",
            func=_parse_and_call_metadata_checker, # Use the new helper function here
            description="Provides information about available data in the database. Can fetch 'all' metadata (default if no specific type is given), 'metrics', 'dimensions', 'product_types', 'service_types', 'regions', or 'time_periods'. Input should be a dictionary with keys 'metadata_type' (string) and optionally 'product_name' (string). Use this when the user asks 'what data do you have', 'what metrics are available', 'what dimensions can I query by', 'what service types for wireline', etc."
        ),
        Tool(
            name="QuestionClassifier",
            func=classify_question_type,
            description="Classifies a user query as 'SIMPLE' (direct metric lookup), 'REASONING' (requiring analysis of relationships/trends), 'METADATA_SERVICE_TYPE:PRODUCT_NAME' (if asking for service types for a specific product), or 'METADATA_AVAILABILITY' (if asking about general data existence/availability). Input is the user's query string."
        )
    ]

    # Agent prompt: Instructs the agent to use the relevance checker and output its result.
    agent_prompt_template = PromptTemplate.from_template("""
    You are an AI assistant whose primary purpose is to answer questions about telecom business metrics and provide available data information.
    You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: A valid Python dictionary literal representing the input to the action.
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer from the tool

    Here are the rules for using your tools:
    1. First, always use the 'TelecomQueryRelevanceChecker' tool to determine if the user's query is RELEVANT to telecom business metrics.
    2. If the 'TelecomQueryRelevanceChecker' returns "IRRELEVANT", your Final Answer should be ONLY "IRRELEVANT". Do not use any other tools.
    3. If the 'TelecomQueryRelevanceChecker' returns "RELEVANT", then analyze the user's query further.
    4. If the RELEVANT query is a question about the availability or scope of data (e.g., "hi", "help", "what data is available", "what metrics do you have", "what dimensions can I query by", "what regions data we have", "do we have data for X month", "do you have data for Y product"):
        a. Use the 'QuestionClassifier' tool to get the precise classification.
        b. If 'QuestionClassifier' returns "METADATA_SERVICE_TYPE:PRODUCT_NAME":
            Thought: The query is asking for service types for a specific product. I need to call AvailableMetadataChecker with metadata_type 'service_types' and the extracted product name.
            Action: AvailableMetadataChecker
            Action Input: {{"metadata_type": "service_types", "product_name": "[EXTRACTED_PRODUCT_NAME_LOWERCASE]"}}
            Observation: [Result from AvailableMetadataChecker]
            Thought: I have the service types for the specific product. I can now provide the final answer.
            Final Answer: [Result from AvailableMetadataChecker]
        c. Else if 'QuestionClassifier' returns "METADATA_AVAILABILITY":
            Thought: The query is asking about general data availability for a specific time or product, or general data existence. I should use the 'AvailableMetadataChecker' tool to provide a comprehensive answer about what data is available.
            Action: AvailableMetadataChecker
            Action Input: {{"metadata_type": "all"}}
            Observation: [Result from AvailableMetadataChecker]
            Thought: I have retrieved the available data information. I can now provide the final answer.
            Final Answer: [Result from AvailableMetadataChecker]
        d. For other general metadata queries (e.g., "what metrics", "what dimensions"):
            Thought: The query is asking for general metadata. I need to call AvailableMetadataChecker with the appropriate metadata_type.
            Action: AvailableMetadataChecker
            Action Input: {{"metadata_type": "[DETERMINED_METADATA_TYPE]"}}
            Observation: [Result from AvailableMetadataChecker]
            Thought: I have the available metadata. I can now provide the final answer.
            Final Answer: [Result from AvailableMetadataChecker]
    5. If the RELEVANT query is a specific question about metrics or KPIs that requires fetching data (e.g., "what is the churn rate", "how is ARPU performing"), then you must use the 'QuestionClassifier' tool to classify if it is a "SIMPLE" or "REASONING" question.
        - If 'QuestionClassifier' returns "SIMPLE":
            Thought: The query is a simple metric lookup. I should signal the main application to handle this.
            Final Answer: SIMPLE_QUERY
        - If 'QuestionClassifier' returns "REASONING":
            Thought: The query requires reasoning and relationship analysis. I should signal the main application to handle this.
            Final Answer: REASONING_QUERY

    Question: {input}
    {agent_scratchpad}
    """)

    # Create the LangChain ReAct Agent
    agent = create_react_agent(agent_llm, tools, agent_prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=5)
else:
    st.error("LangChain agent could not be initialized due to missing or invalid configurations.")
    agent_executor = None # Ensure agent_executor is None if not initialized


# --- Main Execution Flow ---
try:
    if user_query:
        if not agent_executor:
            st.error("Cannot process query: Agent is not initialized due to configuration issues. Please check your config.py.")
            # Append user message, but no assistant response for this case
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            st.rerun() # Rerun to display error and user message

        with st.spinner("Processing your query..."):
            try:
                # --- Input Guardrail: Validate Query Relevance and Intent using LangChain Agent ---
                agent_response = agent_executor.invoke({"input": user_query})

                # The agent's final output determines the next step
                agent_decision = agent_response.get('output', '').strip()

                current_answer = ""
                current_contexts = [] # To store the raw data for Ragas contexts
                ground_truth_answer = "" # Initialize ground truth
                cypher_query = ""
                cypher_metrics = {}

                # Load metrics (moved here for general availability)
                metrics_file_path = os.path.join("config", "metrics.json")
                metric_names = []
                try:
                    with open(metrics_file_path, "r") as f:
                        metrics_data = json.load(f)
                    metric_names = list(metrics_data.keys())
                except Exception as e:
                    st.error(f"Failed to load metrics.json. Ensure it exists in the config directory. Error: {e}")
                    traceback.print_exc()
                    # If metrics are critical for a subsequent step, you might need to handle this more strictly

                if agent_decision == "IRRELEVANT":
                    current_answer = "I can only provide information related to telecom business metrics and KPIs. Please rephrase your question to be about these topics."
                    # No Ragas data collected for irrelevant queries
                    ground_truth_answer = current_answer # For irrelevant queries, the canned response is the ground truth.


                elif agent_decision.startswith("METADATA_SERVICE_TYPE:"):
                    # Extract product name from the agent decision
                    product_name_for_services = agent_decision.split(":")[1].strip()
                    # Capture the raw metadata results here
                    current_answer, cypher_query, cypher_metrics, raw_metadata_results = get_available_metadata_from_neo4j(
                        metadata_type="service_types",
                        product_name=product_name_for_services
                    )
                    ground_truth_answer = current_answer # The metadata output is the ground truth.
                    current_contexts = raw_metadata_results # Pass the raw metadata as contexts

                elif agent_decision == "METADATA_AVAILABILITY":
                    # Capture the raw metadata results here
                    current_answer, cypher_query, cypher_metrics, raw_metadata_results = get_available_metadata_from_neo4j(
                        metadata_type="all"
                    )
                    ground_truth_answer = current_answer # The metadata output is the ground truth.
                    current_contexts = raw_metadata_results # Pass the raw metadata as contexts

                elif agent_decision == "SIMPLE_QUERY" or agent_decision == "REASONING_QUERY":
                    parsed = get_relevant_metrics_and_relationships(user_query, metric_names)
                    extracted_metrics = parsed.get('metrics', [])

                    if not extracted_metrics or not isinstance(extracted_metrics, list) or not any(extracted_metrics):
                        current_answer = "I couldn't extract any relevant telecom metrics from your question. Please make sure your query explicitly mentions metrics I can understand."
                        ground_truth_answer = "No telecom metrics extracted from the query."
                    else:
                        if agent_decision == "SIMPLE_QUERY":
                            cypher_query = generate_simple_cypher(parsed)
                            print(f"Generated SIMPLE Cypher Query: {cypher_query}") # Debugging
                        else: # REASONING_QUERY
                            cypher_query = generate_cypher(parsed)
                            print(f"Generated REASONING Cypher Query: {cypher_query}") # Debugging

                        if cypher_query:
                            results, cypher_metrics = query_neo4j(cypher_query)
                            # Convert each dictionary item into a more natural language string for Ragas contexts
                            current_contexts = []
                            for item in results:
                                readable_parts = []
                                for k, v in item.items():
                                    val_str = str(v) if v is not None else "N/A"
                                    readable_parts.append(f"{k.replace('_', ' ').title()}: {val_str}")
                                current_contexts.append("; ".join(readable_parts))

                            if results:
                                ground_truth_answer = generate_ground_truth_from_data(user_query, results)
                                if agent_decision == "SIMPLE_QUERY":
                                    story = build_simple_answer(results, user_query)
                                else: # REASONING_QUERY
                                    story = build_narrative(results, user_query)

                                # --- Answer Validation Guardrail ---
                                validation_status = validate_answer_with_ai(user_query, story, results)

                                if "SUCCESS" in validation_status:
                                    current_answer = story
                                else:
                                    st.warning(f"The generated answer did not pass validation. Validation Notes: {validation_status}")
                                    current_answer = f"I encountered an issue generating a fully validated answer for your query. Here's what I was able to gather: \n\n{story}\n\n*Validation Note: {validation_status}*"
                            else:
                                current_answer = "No relevant data found in the graph based on your query and extracted entities. Try a different query or ensure data exists for these parameters."
                                ground_truth_answer = "No data found for this query based on extracted parameters."
                        else:
                            current_answer = "I couldn't generate a valid Cypher query from your request."
                            ground_truth_answer = "Could not generate a Cypher query from the request."

                else:
                    # This block is reached when agent_decision is a direct Final Answer from the agent
                    # (e.g., a metadata query response like "We have data for the following regions...")
                    current_answer = agent_decision # The agent's final answer is the message to display.

                    # Determine metadata_type_requested from the original user query to re-fetch cypher_query and cypher_metrics
                    metadata_type_requested = "all" # Default
                    if "regions" in user_query.lower():
                        metadata_type_requested = "regions"
                    elif "dimensions" in user_query.lower():
                        metadata_type_requested = "dimensions"
                    elif "metrics" in user_query.lower():
                        metadata_type_requested = "metrics"
                    elif "product type" in user_query.lower() or "products" in user_query.lower():
                        metadata_type_requested = "product_types"
                    elif "service type" in user_query.lower() or "services" in user_query.lower():
                        # This case is now handled by the specific "METADATA_SERVICE_TYPE:" branch above.
                        # This 'else' block will primarily catch general "all" or "dimensions" requests.
                        metadata_type_requested = "service_types" # Fallback, though less specific.
                    elif "time period" in user_query.lower() or "months" in user_query.lower() or "years" in user_query.lower():
                        metadata_type_requested = "time_periods"
                    elif "hi" in user_query.lower() or "help" in user_query.lower() or "what data is available" in user_query.lower():
                        metadata_type_requested = "all"
                    
                    # Re-call get_available_metadata_from_neo4j to get the associated Cypher query and metrics
                    # (The first return value is ignored as current_answer is already set from agent_decision)
                    _, cypher_query, cypher_metrics, raw_metadata_results = get_available_metadata_from_neo4j(metadata_type_requested)
                    ground_truth_answer = current_answer # For metadata questions, the answer itself can serve as ground truth.
                    current_contexts = raw_metadata_results # Pass the raw metadata as contexts

                # Append user message
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                # Store cypher_query and cypher_metrics with the assistant's message
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": current_answer, # Use current_answer which holds the final message
                    "cypher_query": cypher_query if cypher_query else None,
                    "cypher_metrics": cypher_metrics if cypher_metrics else None
                })
                # Collect data for Ragas evaluation
                st.session_state.evaluation_data.append({
                    "question": user_query,
                    "answer": current_answer,
                    "contexts": current_contexts, # This now contains raw metadata for metadata queries
                    "ground_truth": ground_truth_answer, # Pass the generated ground truth
                })
                st.rerun() # Rerun immediately after updating chat history

            except Exception as e:
                st.error(f"An unexpected error occurred during agent processing: {e}")
                current_answer = f"An unexpected error occurred: {e}"
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                st.session_state.chat_history.append({"role": "assistant", "content": current_answer})
                # On error, log for Ragas with an error message in ground_truth
                st.session_state.evaluation_data.append({
                    "question": user_query,
                    "answer": current_answer,
                    "contexts": [],
                    "ground_truth": f"Error occurred during processing: {e}"
                })
                st.rerun() # Rerun even on error to display the error message

except Exception as e:
    st.error(f"An error occurred while processing your request: {e}")
    traceback.print_exc()
