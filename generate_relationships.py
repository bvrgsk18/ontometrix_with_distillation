import pandas as pd
import csv
import json
import os
import sys
import uuid
# Import the Google Generative AI client library
import google.generativeai as genai
import argparse

# --- RAGAS Imports and Configuration ---
from datasets import Dataset # RAGAS uses Hugging Face Datasets
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
# Import RAGAS's base LLM classes for proper integration
from ragas.llms.base import BaseRagasLLM
from typing import NamedTuple, Optional, Dict, Any, List
import asyncio

# --- LangChain Imports for Embeddings and Generation Objects ---
# Use the compatible LangChain Google Generative AI embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from langchain_core.outputs import Generation, LLMResult # Import Generation and LLMResult

# --- Configuration ---
# Get API key from environment variable or local config
GEMINI_API_KEY_VALUE = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY_VALUE is None:
    try:
        from config import GEMINI_API_TOKEN
        GEMINI_API_KEY_VALUE = GEMINI_API_TOKEN
    except ImportError:
        print("Warning: config.py not found or GEMINI_API_TOKEN not set. Ensure GEMINI_API_KEY is set as an environment variable.")

if GEMINI_API_KEY_VALUE is None:
    print("Error: GEMINI_API_KEY or GEMINI_API_TOKEN not found. Please set it as an environment variable or in config.py.")
    sys.exit(1)

# Set the API key for genai
genai.configure(api_key=GEMINI_API_KEY_VALUE)

# Define the model to use for RAGAS if it's not already defined in config.
if 'RELATIONSHIPS_LLM_MODEL' not in locals():
    RELATIONSHIPS_LLM_MODEL = "gemini-1.5-flash" # Default to a suitable model

# --- RAGAS LLM Wrapper for Gemini ---
class GeminiRagasLLM(BaseRagasLLM):
    def __init__(self, model_name: str):
        self.model = genai.GenerativeModel(model_name)
        super().__init__() # Call the constructor of the base class

    def generate_text(self, prompt: str | List, **kwargs) -> LLMResult:
        """
        Synchronously generates content using the Gemini model.
        The prompt can be a string or a list of Langchain-like messages.
        Returns an LLMResult object expected by RAGAS, which contains a 'generations' attribute.
        """
        # Convert list of messages to a single string if needed for Gemini's API
        if isinstance(prompt, List):
            # Assuming prompt is a list of Langchain BaseMessage objects
            prompt_text = "\n".join([str(p.content) if hasattr(p, 'content') else str(p) for p in prompt])
        else:
            prompt_text = str(prompt)

        # Merge any generation config from kwargs (e.g., temperature, response_mime_type)
        current_generation_config = kwargs.get('generation_config', {})
        
        try:
            response = self.model.generate_content(
                prompt_text,
                generation_config=current_generation_config
            )
            # Extract text from the response
            generated_text = ""
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                generated_text = response.candidates[0].content.parts[0].text
            
            # RAGAS expects an LLMResult with a 'generations' attribute
            # 'generations' is a list of lists of Generation objects
            generations = [[Generation(text=generated_text)]]

            # Create llm_output for token usage if available
            llm_output = {}
            if hasattr(response, 'usage_metadata') and response.usage_metadata is not None:
                llm_output = {"token_usage": {
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count
                }}

            return LLMResult(generations=generations, llm_output=llm_output)
        except Exception as e:
            print(f"Error in GeminiRagasLLM generate_text: {e}")
            # Return an LLMResult with empty generations on error
            return LLMResult(generations=[[Generation(text="", generation_info={"error": str(e)})]], llm_output={})


    async def agenerate_text(self, prompt: str | List, **kwargs) -> LLMResult:
        """
        Asynchronously generates content using the Gemini model.
        For simplicity, this calls the synchronous method within an async context.
        For true asynchronous behavior, you would use an async Gemini client if available.
        """
        return await asyncio.to_thread(self.generate_text, prompt, **kwargs)


# Initialize RAGAS LLM
ragas_llm = GeminiRagasLLM(RELATIONSHIPS_LLM_MODEL) # Use the same model or a dedicated one for RAGAS

# Initialize Embeddings for RAGAS, passing the API key explicitly
ragas_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY_VALUE) 

# --- Define Relationship Types ---
def define_relationship_types(relationships_file):
    """
    Loads the allowed types of relationships between metrics from a JSON file.
    These phrases are used to guide the LLM in generating relationships.
    """
    with open(relationships_file, 'r', encoding='utf-8') as file:
        relationships = json.load(file)
    return relationships

# --- Extract Unique Metrics from CSV ---
def get_distinct_metrics_from_csv(file_path, metric_column='metric_name'):
    """
    Reads a CSV file and extracts a list of unique metric names from a specified column.

    Args:
        file_path (str): The path to the CSV file.
        metric_column (str): The name of the column containing metric names.

    Returns:
        list: A list of unique metric names. Returns an empty list if an error occurs
              or the column is not found.
    """
    try:
        df = pd.read_csv(file_path)
        # Strip whitespace from column names to ensure accurate matching
        df.columns = df.columns.str.strip()
        if metric_column not in df.columns:
            print(f"Missing column '{metric_column}' in CSV.")
            return []
        # Return unique, non-null metric names
        return df[metric_column].dropna().unique().tolist()
    except Exception as e:
        print(f"CSV Read Error: {e}")
        return []

# --- Load Metric Definitions from JSON ---
def load_metric_definitions(file_path):
    """
    Loads metric definitions (including direction and description) from a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: A dictionary where keys are metric names and values are their definitions.
              Returns an empty dictionary if the file is not found or decoding fails.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: metrics.json file not found at {file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}. Check file format.")
        return {}
    except Exception as e:
        print(f"Error loading metric definitions: {e}")
        return {}

# --- Use Gemini to Discover Metric Relationships ---
def get_relationships_from_gemini(metrics_info, allowed_relationship_phrases):
    """
    Uses an LLM (Gemini 2.5 Flash) to identify relationships between telecom KPIs based on
    their descriptions and desired directions.

    Args:
        metrics_info (dict): A dictionary mapping metric names to their details
                             (direction, description).
        allowed_relationship_phrases (list): A list of phrases defining valid relationship types.

    Returns:
        tuple: A tuple containing:
            - list: A list of dictionaries, where each dictionary represents a discovered
                    relationship (metric_a, relationship_type, metric_b, reasoning).
                    Returns an empty list if the API call fails or no relationships are found.
            - str: The full prompt string used for generation, to be used as 'question' for RAGAS.
            - str: The metric details string used as 'context' for RAGAS.
    """
    metric_details_str = ""
    # Format metric details for the LLM prompt
    for metric_name, details in metrics_info.items():
        direction = details.get('direction', 'N/A')
        description = details.get('description', 'No description provided.')
        metric_details_str += f"- {metric_name} (Direction: {direction}, Description: {description})\n"

    # Construct the prompt for relationship generation
    full_prompt = f"""
    You are a Telecom Business Analyst. Identify relationships among these KPIs. For each KPI, its desired direction (high or low value better) and a brief description are provided to help you understand its context and impact.
    metrics can have multiple relationships with other metrics.
    Detect most suitable relationships.
    try to define relationships all for metrics.
    In general Sales related call volume will help to increase gross adds.
    In general Service related call volume , trouble tickets will positively impact wirless disconnects and increases churn rate and decreases NPS score and qes score
    Net adds will increase ARPU.
    For metrics having direction low value is better then use relation ships either "increases the value of" or "decreases the value of" (e.g., "Wireless Disconnects, increases the value of, Wireless Churn Rate", "Wireless Net Adds, decreases the value of, Wireless Churn Rate").
    
    There are groups of metrics, e.g., "Wireless Net Adds*", "Wireless Disconnects*", etc. We need to include an additional relationship as "related metrics" if not detected by you.
    For example, the relationships below are DIRECT and ALLOWED:
    1) Wireless Net Adds,decreases the value of,Wireless Churn Rate,"More net adds often reflect improved retention, reducing churn rate."
    2) Wireless Port Out,increases the value of,Wireless Churn Rate,"More port outs lead to higher churn."
    3) Number Of Customers with Autopay Discount,decreases the value of,Wireless Churn Rate,"Autopay customers are more stable, reducing churn."
    4) Number Of Customers with Autopay Discount,decreases the value of,Wireless Port Out,"Autopay users are more loyal, less likely to port out."
    5) NPS Score,inversely proportional to,Wireless Churn Rate,"High NPS reflects satisfaction, leading to lower churn. Lower NPS score indicates dissatisfaction, leading to higher churn."
    6) Average Call handling Time - Service,directly proportional to,Wireless Churn Rate,"If average call handlgling for service related call increases , customer dissatifcation increases and churn rate increases."
    For example, the relationships below are NOT ALLOWED (indirect or incorrect causality):
    1) ARPU,increases the value of,Wireless Net Adds,"Higher ARPU can lead to increased investment in acquiring new wireless subscribers, thus increasing net adds."
    2) Wireless Churn Rate,negatively impacts,Wireless Net Adds,Higher churn rate results in a decrease in the net adds of wireless subscribers.
    3) NPS Score,increases the value of,Wireless Net Adds,"Higher NPS scores indicate satisfied customers who are more likely to recommend the service, leading to potential growth in net adds."

The examples below are completely wrong (e.g., how come new customers increase churn rate?):
    1) Wireless Net Adds - New customers,increases the value of,Wireless Churn Rate,Higher net adds of new customers may indicate better retention and lower churn rate.
    2) Wireless Net Adds - Add a Line (AAL),increases the value of,Wireless Churn Rate,More AALs by existing customers may indicate satisfaction and lower likelihood of churn.

If KPIs are clearly part of the same family (e.g., share a common prefix like "Wireless Net Adds", "Wireless Disconnects"), also link them as "related metrics" if no direct relationship is detected.
Metrics with Details:
{metric_details_str}

Allowed relationship types:
{chr(10).join(['- ' + r for r in allowed_relationship_phrases])}

Output format:
[
  {{
    "metric_a": "...",
    "relationship_type": "...",
    "metric_b": "...",
    "reasoning": "..."
  }}
]"""

    try:
        model = genai.GenerativeModel(RELATIONSHIPS_LLM_MODEL)
        response = model.generate_content(
            full_prompt,
            generation_config={
                "temperature": 0.3,
                "response_mime_type": "application/json",
                "response_schema": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "metric_a": {"type": "STRING"},
                            "relationship_type": {"type": "STRING"},
                            "metric_b": {"type": "STRING"},
                            "reasoning": {"type": "STRING"}
                        },
                        "required": ["metric_a", "relationship_type", "metric_b", "reasoning"]
                    }
                }
            }
        )
        
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            reply_json_str = response.candidates[0].content.parts[0].text
            return json.loads(reply_json_str), full_prompt, metric_details_str
        else:
            print("Gemini API Error: No content found in the response.")
            return [], full_prompt, metric_details_str
    except Exception as e:
        print(f"Gemini API Error (get_relationships_from_gemini): {e}")
        return [], full_prompt, metric_details_str

# --- NEW: Use Gemini to Validate Metric Relationships ---
def validate_relationships_with_gemini(relationships_to_validate, metrics_info):
    """
    Validates a list of generated metric relationships using a second LLM call (Gemini 2.5 Flash).
    The LLM acts as a critical reviewer, ensuring logical soundness and direct impact.

    Args:
        relationships_to_validate (list): A list of dictionaries, each representing
                                          a relationship to be validated.
        metrics_info (dict): A dictionary mapping metric names to their details
                             (direction, description), providing context for validation.

    Returns:
        list: A list of dictionaries representing the relationships that passed validation.
              Invalid relationships are omitted from the returned list.
    """
    if not relationships_to_validate:
        print("No relationships to validate.")
        return []

    metric_details_str = ""
    for metric_name, details in metrics_info.items():
        direction = details.get('direction', 'N/A')
        description = details.get('description', 'No description provided.')
        metric_details_str += f"- {metric_name} (Direction: {direction}, Description: {description})\n"

    relationships_str = json.dumps(relationships_to_validate, indent=2)

    full_prompt = f"""You are a senior Telecom Business Analyst with extensive domain knowledge.
Your task is to critically review and validate a list of proposed relationships between Key Performance Indicators (KPIs).

Here are the details for each metric, including its desired direction (high or low value is better) and a brief description:
{metric_details_str}

Here are the relationships you need to validate:
{relationships_str}

For each relationship, assess its logical correctness and direct impact based on:
1.  The provided 'direction' and 'description' of Metric A and Metric B.
2.  Your general telecom domain knowledge regarding products (sales, service, support, etc.).
3.  Ensure the relationship describes a DIRECT impact, not an indirect one.
4.  Correct any relationships that imply wrong causality (e.g., "new customers increase churn rate" is wrong).

Return ONLY the relationships that are logically VALID and DIRECTLY IMPACTFUL.
Maintain the exact same JSON array format as provided for the input relationships.
If a relationship is invalid, simply omit it from the output.

Output format:
[
  {{
    "metric_a": "...",
    "relationship_type": "...",
    "metric_b": "...",
    "reasoning": "..."
  }},
  ...
]"""

    try:
        model = genai.GenerativeModel(RELATIONSHIPS_LLM_MODEL)
        response = model.generate_content(
            full_prompt,
            generation_config={
                "temperature": 0.2,
                "response_mime_type": "application/json",
                "response_schema": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "metric_a": {"type": "STRING"},
                            "relationship_type": {"type": "STRING"},
                            "metric_b": {"type": "STRING"},
                            "reasoning": {"type": "STRING"}
                        },
                        "required": ["metric_a", "relationship_type", "metric_b", "reasoning"]
                    }
                }
            }
        )
        
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            reply_json_str = response.candidates[0].content.parts[0].text
            return json.loads(reply_json_str)
        else:
            print("Gemini API Error: No content found in the response.")
            return []
    except Exception as e:
        print(f"Gemini API Error (validate_relationships_with_gemini): {e}")
        return []

# --- Save Relationships to CSV ---
def save_relationships_to_csv(relationships, output_file):
    """
    Saves a list of relationships to a CSV file.

    Args:
        relationships (list): A list of dictionaries, each representing a relationship.
        output_file (str): The path to the output CSV file.
    """
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['metric_a', 'relationship_type', 'metric_b', 'reasoning'])
            writer.writeheader()
            writer.writerows(relationships)
        print(f"Successfully saved {len(relationships)} relationships to {output_file}")
    except Exception as e:
        print(f"CSV Write Error: {e}")

# --- NEW: Load Relationships from CSV (for Ground Truth) ---
def load_relationships_from_csv(file_path):
    """
    Loads relationships from a CSV file into a list of dictionaries.
    Expected CSV columns: 'metric_a', 'relationship_type', 'metric_b', 'reasoning'.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        list: A list of dictionaries, each representing a relationship.
              Returns an empty list if the file is not found or reading fails.
    """
    relationships = []
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                relationships.append({
                    'metric_a': row.get('metric_a', '').strip(),
                    'relationship_type': row.get('relationship_type', '').strip(),
                    'metric_b': row.get('metric_b', '').strip(),
                    'reasoning': row.get('reasoning', '').strip()
                })
        print(f"Successfully loaded {len(relationships)} ground truth relationships from {file_path}")
    except FileNotFoundError:
        print(f"Warning: Ground truth file not found at {file_path}. RAGAS evaluation may be skipped if no ground truth is available.")
        return []
    except Exception as e:
        print(f"Error loading ground truth relationships from CSV: {e}")
        return []
    return relationships

# --- RAGAS Evaluation Function ---
def run_ragas_evaluation(questions: str, contexts: str, generated_answers: List[Dict[str, str]], ground_truths: List[Dict[str, str]]):
    """
    Runs RAGAS evaluation on the generated relationships against ground truth.

    Args:
        questions (str): The main prompt used to generate relationships.
        contexts (str): The metric definitions used as context for generation.
        generated_answers (List[Dict[str, str]]): The relationships generated by the LLM.
        ground_truths (List[Dict[str, str]]): The gold standard relationships.
    """
    if not generated_answers or not ground_truths:
        print("\nSkipping RAGAS evaluation: Not enough data (generated relationships or ground truths) to evaluate.")
        return

    generated_answers_str = json.dumps(generated_answers, indent=2)
    ground_truths_str = json.dumps(ground_truths, indent=2)

    data = {
        'question': [questions],
        'answer': [generated_answers_str],
        'contexts': [[contexts]],
        'ground_truth': [ground_truths_str]
    }

    dataset = Dataset.from_dict(data)

    print("\n--- Running RAGAS Evaluation ---")
    metrics = [
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    ]

    for metric in metrics:
        metric.llm = ragas_llm
        # Explicitly set embeddings for metrics that require them.
        # context_recall and context_precision often rely on embeddings.
        if hasattr(metric, 'embeddings'):
            metric.embeddings = ragas_embeddings 

    try:
        result = evaluate(
            dataset,
            metrics=metrics,
            llm=ragas_llm, # Pass the custom GeminiRagasLLM
            embeddings=ragas_embeddings # Pass the custom GoogleGenerativeAIEmbeddings
        )
        print("RAGAS Evaluation Results:")
        print(result)
        # Save RAGAS evaluation results to CSV
        result_df = result.to_pandas()
        ragas_output_csv = "data/ragas_evaluation_results.csv"
        try:
            result_df.to_csv(ragas_output_csv, index=False)
            print(f"RAGAS evaluation results saved to {ragas_output_csv}")
        except Exception as e:
            print(f"Error saving RAGAS results to CSV: {e}")
    except Exception as e:
        print(f"Error during RAGAS evaluation: {e}")
        print("Please ensure your RAGAS and its dependencies are correctly installed and configured.")
        print("Specifically, check if you have an embedding model set up if required by the metrics you are using.")


# --- Main Execution Block ---
if __name__ == "__main__":
    data_folder = "data"
    os.makedirs(data_folder, exist_ok=True)
    config_folder = "config"
    os.makedirs(config_folder, exist_ok=True)

    parser = argparse.ArgumentParser(description="Generate and validate telecom metric relationships using Gemini.")
    parser.add_argument('--test', action='store_true', help='Use test data files and output to a test-specific CSV.')
    args = parser.parse_args()

    if args.test:
        csv_path = "data/test_generated_telecom_data_all.csv"
        defs_file = "config/metrics.json"
        relationships_file = "config/relationships.json"
        output_csv = "data/test_telecom_metric_relationships.csv"
        relationships_ground_truth_path = "data/gold_telecom_metric_relationships.csv"
    else:
        csv_path = "data/generated_telecom_data_all.csv"
        defs_file = "config/metrics.json"
        relationships_file = "config/relationships.json"
        output_csv = "data/telecom_metric_relationships.csv"
        relationships_ground_truth_path = None



    metric_definitions = load_metric_definitions(defs_file)
    if not metric_definitions:
        sys.exit("Could not load metric definitions from metrics.json. Please ensure the file exists and is correctly formatted. Exiting.")

    distinct_metrics_from_csv = get_distinct_metrics_from_csv(csv_path)
    if not distinct_metrics_from_csv:
        sys.exit("No metrics found in the CSV file. Please ensure the CSV contains data in the 'metric_name' column. Exiting.")

    metrics_for_llm = {}
    for metric_name in distinct_metrics_from_csv:
        if metric_name in metric_definitions:
            metrics_for_llm[metric_name] = metric_definitions[metric_name]
        else:
            metrics_for_llm[metric_name] = {"direction": "N/A", "description": "No detailed definition found."}

    print(f"Extracted {len(metrics_for_llm)} metrics with details from '{csv_path}' and '{defs_file}'.")
    print("Initiating relationship generation using Gemini...")

    relationship_phrases = list(define_relationship_types(relationships_file).values())

    relationships, generation_prompt, generation_context_str = get_relationships_from_gemini(metrics_for_llm, relationship_phrases)

    validated_relationships = []
    if relationships:
        print(f"Successfully generated {len(relationships)} initial relationships.")
        print("Proceeding to validate the generated relationships using another Gemini call...")

        validated_relationships = validate_relationships_with_gemini(relationships, metrics_for_llm)

        if validated_relationships:
            save_relationships_to_csv(validated_relationships, output_csv)
            print(f"Process completed successfully. Validated relationships saved to '{output_csv}'.")
        else:
            print("No relationships passed the validation step.")
    else:
        print("No relationships were generated in the initial step.")

    print("Overall process finished.")

    print("\n--- Metric Relationship Report ---")

    total_metrics_in_json = len(metric_definitions)
    print(f"Total number of metrics in metrics.json: {total_metrics_in_json}")

    metrics_with_data_in_csv = len(distinct_metrics_from_csv)
    print(f"Number of metrics with data in the CSV file: {metrics_with_data_in_csv}")

    metrics_with_relationships_set = set()
    if validated_relationships:
        for rel in validated_relationships:
            metrics_with_relationships_set.add(rel.get('metric_a'))
            metrics_with_relationships_set.add(rel.get('metric_b'))
    num_metrics_with_relationships = len(metrics_with_relationships_set)
    print(f"Number of unique metrics involved in relationships: {num_metrics_with_relationships}")

    metrics_with_some_details = set(metrics_for_llm.keys())
    metrics_without_relationships_count = len(metrics_with_some_details) - num_metrics_with_relationships
    print(f"Number of metrics with details (from CSV or JSON) but no defined relationships: {metrics_without_relationships_count}")

    print("Metrics without defined relationships:")
    for metric_name in metrics_with_some_details:
        if metric_name not in metrics_with_relationships_set:
            print(f"- {metric_name}")
    print("--- Report End ---")

    if args.test:
        ground_truths_from_file = load_relationships_from_csv(relationships_ground_truth_path)
        
        run_ragas_evaluation(
            questions=generation_prompt,
            contexts=generation_context_str,
            generated_answers=relationships,
            ground_truths=ground_truths_from_file
        )