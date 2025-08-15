import uuid
import random
import datetime
import pandas as pd
import argparse
import json
import os
import itertools # For generating combinations
import numpy as np # For pd.notna

# Ensure config directory exists and create dummy config files if they don't
config_dir = "config"
os.makedirs(config_dir, exist_ok=True)

# Create dummy config files if they don't exist
# regions.json
regions_path = os.path.join(config_dir, "regions.json")

# products.json
products_path = os.path.join(config_dir, "products.json")

# metrics.json
metrics_path = os.path.join(config_dir, "metrics.json")

# unique_columns.json
unique_columns_path = os.path.join(config_dir, "unique_columns.json")


# Load configurations
class Config:
    def __init__(self):
        self.regions = self._load_config("regions.json")
        self.products = self._load_config("products.json")
        self.metrics_info = self._load_config("metrics.json")
        self.unique_columns = self._load_config("unique_columns.json")

    def _load_config(self, filename):
        with open(os.path.join("config", filename), "r") as f:
            return json.load(f)

config = Config()

def get_quarter_name(month: int) -> str:
    """Returns the quarter name for a given month."""
    if 1 <= month <= 3:
        return "Q1"
    elif 4 <= month <= 6:
        return "Q2"
    elif 7 <= month <= 9:
        return "Q3"
    elif 10 <= month <= 12:
        return "Q4"
    else:
        raise ValueError("Month must be between 1 and 12")

def get_rollup_function(metric_name: str, metrics_info: dict) -> str:
    """
    Returns the appropriate pandas aggregation function string ('sum' or 'mean')
    based on the 'rollup' property in metrics_info. Defaults to 'sum'.
    """
    return metrics_info.get(metric_name, {}).get('rollup', 'sum')

def generate_data_for_month(year: int, month: int, target_rows_per_month=5000):
    """Generates only monthly data for a given year and month."""
    data = []
    load_ts = datetime.datetime.now().isoformat()
    rpt_mth_monthly = f"{datetime.date(1900, month, 1).strftime('%B')}"
    rpt_year = str(year)
    
    print(f"Generating monthly data for {rpt_mth_monthly}...")

    # Generate all valid combinations based on products/services/metrics
    possible_combinations = []    
    for metric_name, metric_spec in config.metrics_info.items():
        allowed_products = metric_spec.get('product_name', list(config.products.keys()))
        if isinstance(allowed_products, str):
            allowed_products = allowed_products.split(',') # Handle comma-separated products
        
        # Determine which products to iterate based on if 'product_name' is specified in metric_spec
        products_to_iterate = [p for p in allowed_products if p in config.products] if 'product_name' in metric_spec else config.products.keys()

        for product_name in products_to_iterate:
            # Ensure the product_name from the metric_spec actually exists in config.products
            if product_name not in config.products:
                continue

            allowed_service_types = config.products.get(product_name, [])
            for service_type in allowed_service_types:
                for region in config.regions:
                    possible_combinations.append(
                        (product_name, service_type, region, rpt_mth_monthly, rpt_year, metric_name)
                    )

    num_rows_to_generate = min(target_rows_per_month, len(possible_combinations))
    
    if num_rows_to_generate == 0 and target_rows_per_month > 0:
        print(f"Warning: No possible unique combinations for {rpt_mth_monthly}. Skipping data generation for this month.")
        return pd.DataFrame(data)

    if len(possible_combinations) == 0:
        print(f"No possible combinations for {rpt_mth_monthly}. Returning empty DataFrame.")
        return pd.DataFrame(data)

    # Sample combinations to meet target_rows_per_month
    sampled_combinations = random.sample(possible_combinations, num_rows_to_generate)

    # Dictionary to hold generated 'Wireless' metrics to enforce business rules
    wireless_metrics_data = {}

    # Define core wireless metrics involved in business rules
    core_wireless_metrics = [
        "Wireless Gross Adds",
        "Wireless Disconnects",
        "Wireless Net Adds",
        "Wireless Net Adds - Add a Line (AAL)",
        "Wireless Net Adds - New customers"
    ]

    # Define new metric groups for business rules
    customer_trouble_tickets_parent = "Customer Trouble Tickets Count"
    customer_trouble_tickets_components = [
        "Customer Trouble Tickets - Call Drop",
        "Customer Trouble Tickets - Discount Related",
        "Customer Trouble Tickets - Network Coverage",
        "Customer Trouble Tickets - Slow Net Speed"
    ]

    customer_interactions_parent = "Number of Customer Interactions"
    customer_interactions_components = [
        "Number of Customer Interactions - App",
        "Number of Customer Interactions - Call Centers",
        "Number of Customer Interactions - Portal",
        "Number of Customer Interactions - Store",
        "Number Of Customers with Autopay Discount",
        "Number Of Customers with Late Fee Waiver"
    ]

    # Dictionaries to store pre-calculated metric values for each unique combination
    customer_ticket_data = {}
    customer_interaction_data = {}

    for product_name, service_type, region, rpt_mth, rpt_year, metric_name in sampled_combinations:
        metric_spec = config.metrics_info[metric_name]
        high_or_low_better = metric_spec["direction"]
        
        metric_value = 0 # Default value

        unique_key = (product_name, service_type, region) # Key to ensure consistency across product/service/region

        # Handle Wireless product metrics (existing logic)
        if product_name == "Wireless" and metric_name in core_wireless_metrics:
            if unique_key not in wireless_metrics_data:
                # Generate base Wireless metrics if not already generated for this combination
                wg_adds = random.randint(config.metrics_info["Wireless Gross Adds"]["range"][0], config.metrics_info["Wireless Gross Adds"]["range"][1])
                w_disconnects = random.randint(config.metrics_info["Wireless Disconnects"]["range"][0], config.metrics_info["Wireless Disconnects"]["range"][1])
                
                wn_adds = wg_adds - w_disconnects
                
                # Ensure WN_Adds is not negative, adjusting disconnects if necessary
                if wn_adds < 0:
                    max_allowed_disconnects = wg_adds 
                    w_disconnects = random.randint(config.metrics_info["Wireless Disconnects"]["range"][0], min(config.metrics_info["Wireless Disconnects"]["range"][1], max_allowed_disconnects))
                    wn_adds = wg_adds - w_disconnects
                    if wn_adds < 0: wn_adds = 0 # Final safeguard
                    
                # Distribute WN_Adds into Adda_line and New_customers
                wn_adds_addaline = random.randint(0, wn_adds)
                wn_adds_newcustomers = wn_adds - wn_adds_addaline

                wireless_metrics_data[unique_key] = {
                    "Wireless Gross Adds": wg_adds,
                    "Wireless Disconnects": w_disconnects,
                    "Wireless Net Adds": wn_adds,
                    "Wireless Net Adds - Add a Line (AAL)": wn_adds_addaline,
                    "Wireless Net Adds - New customers": wn_adds_newcustomers
                }
            
            # Assign value from the pre-calculated wireless_metrics_data
            metric_value = wireless_metrics_data[unique_key][metric_name]

        # Handle Customer Trouble Tickets business rule
        elif product_name == "Wireless" and (metric_name == customer_trouble_tickets_parent or metric_name in customer_trouble_tickets_components):
            if unique_key not in customer_ticket_data:
                # Generate parent metric first
                parent_range = config.metrics_info[customer_trouble_tickets_parent]["range"]
                parent_value = random.randint(parent_range[0], parent_range[1])

                # Distribute parent value among components
                temp_component_values = {}
                remaining_value = parent_value
                
                for i, component_metric in enumerate(customer_trouble_tickets_components):
                    comp_range = config.metrics_info[component_metric]["range"]
                    if i < len(customer_trouble_tickets_components) - 1:
                        min_for_others = sum(config.metrics_info[c]["range"][0] for c in customer_trouble_tickets_components[i+1:])
                        max_possible_for_this = min(comp_range[1], remaining_value - min_for_others)
                        
                        if max_possible_for_this < comp_range[0]:
                            comp_val = comp_range[0]
                        else:
                            comp_val = random.randint(comp_range[0], max(comp_range[0], max_possible_for_this))
                    else:
                        comp_val = max(comp_range[0], min(comp_range[1], remaining_value))
                    
                    temp_component_values[component_metric] = comp_val
                    remaining_value -= comp_val
                    if remaining_value < 0: remaining_value = 0

                current_sum = sum(temp_component_values.values())
                difference = parent_value - current_sum

                if difference != 0:
                    if difference > 0:
                        addable_components = [c for c in customer_trouble_tickets_components if temp_component_values[c] < config.metrics_info[c]["range"][1]]
                        if addable_components:
                            random.shuffle(addable_components)
                            for comp in addable_components:
                                add_amount = min(difference, config.metrics_info[comp]["range"][1] - temp_component_values[comp])
                                temp_component_values[comp] += add_amount
                                difference -= add_amount
                                if difference == 0: break
                    elif difference < 0:
                        subtractable_components = [c for c in customer_trouble_tickets_components if temp_component_values[c] > config.metrics_info[c]["range"][0]]
                        if subtractable_components:
                            random.shuffle(subtractable_components)
                            for comp in subtractable_components:
                                subtract_amount = min(abs(difference), temp_component_values[comp] - config.metrics_info[comp]["range"][0])
                                temp_component_values[comp] -= subtract_amount
                                difference += subtract_amount
                                if difference == 0: break
                
                for comp_metric in customer_trouble_tickets_components:
                    comp_range = config.metrics_info[comp_metric]["range"]
                    temp_component_values[comp_metric] = max(comp_range[0], min(comp_range[1], temp_component_values[comp_metric]))

                customer_ticket_data[unique_key] = {customer_trouble_tickets_parent: parent_value}
                customer_ticket_data[unique_key].update(temp_component_values)
            
            metric_value = customer_ticket_data[unique_key][metric_name]

        # Handle Number of Customer Interactions business rule
        elif product_name == "Wireless" and (metric_name == customer_interactions_parent or metric_name in customer_interactions_components):
            if unique_key not in customer_interaction_data:
                # Generate parent metric first
                parent_range = config.metrics_info[customer_interactions_parent]["range"]
                parent_value = random.randint(parent_range[0], parent_range[1])

                # Distribute parent value among components
                temp_component_values = {}
                remaining_value = parent_value

                for i, component_metric in enumerate(customer_interactions_components):
                    comp_range = config.metrics_info[component_metric]["range"]
                    if i < len(customer_interactions_components) - 1:
                        min_for_others = sum(config.metrics_info[c]["range"][0] for c in customer_interactions_components[i+1:])
                        max_possible_for_this = min(comp_range[1], remaining_value - min_for_others)
                        
                        if max_possible_for_this < comp_range[0]:
                            comp_val = comp_range[0]
                        else:
                            comp_val = random.randint(comp_range[0], max(comp_range[0], max_possible_for_this))
                    else:
                        comp_val = max(comp_range[0], min(comp_range[1], remaining_value))
                    
                    temp_component_values[component_metric] = comp_val
                    remaining_value -= comp_val
                    if remaining_value < 0: remaining_value = 0

                current_sum = sum(temp_component_values.values())
                difference = parent_value - current_sum

                if difference != 0:
                    if difference > 0:
                        addable_components = [c for c in customer_interactions_components if temp_component_values[c] < config.metrics_info[c]["range"][1]]
                        if addable_components:
                            random.shuffle(addable_components)
                            for comp in addable_components:
                                add_amount = min(difference, config.metrics_info[comp]["range"][1] - temp_component_values[comp])
                                temp_component_values[comp] += add_amount
                                difference -= add_amount
                                if difference == 0: break
                    elif difference < 0:
                        subtractable_components = [c for c in customer_interactions_components if temp_component_values[c] > config.metrics_info[c]["range"][0]]
                        if subtractable_components:
                            random.shuffle(subtractable_components)
                            for comp in subtractable_components:
                                subtract_amount = min(abs(difference), temp_component_values[comp] - config.metrics_info[comp]["range"][0])
                                temp_component_values[comp] -= subtract_amount
                                difference += subtract_amount
                                if difference == 0: break
                
                for comp_metric in customer_interactions_components:
                    comp_range = config.metrics_info[comp_metric]["range"]
                    temp_component_values[comp_metric] = max(comp_range[0], min(comp_range[1], temp_component_values[comp_metric]))

                customer_interaction_data[unique_key] = {customer_interactions_parent: parent_value}
                customer_interaction_data[unique_key].update(temp_component_values)

            metric_value = customer_interaction_data[unique_key][metric_name]

        # For other metrics not part of specific business rules, generate randomly
        else:
            if metric_spec["datatype"] == "int":
                metric_value = random.randint(metric_spec["range"][0], metric_spec["range"][1])
            else: # float
                metric_value = round(random.uniform(metric_spec["range"][0], metric_spec["range"][1]), 2)

        row = {
            "product_name": product_name,
            "service_type": service_type,
            "region": region,
            "rpt_mth": rpt_mth,
            "rpt_year": rpt_year,
            "high_or_low_better": high_or_low_better,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "data_type": "Monthly",
            "load_ts": load_ts
        }
        row["summary"] = f"""{product_name} {service_type} service in {region} for {rpt_mth} {rpt_year} has a metric of {metric_name} with a value of {metric_value}, where {high_or_low_better}"""
        data.append(row)
    
    df_month = pd.DataFrame(data)
    return df_month

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True, help='Year of data to generate')
    parser.add_argument('--rows', type=int, default=5000, help='Number of rows to generate (target per month for monthly data)')
    args = parser.parse_args()

    current_date = datetime.date.today()
    current_year = current_date.year
    current_month = current_date.month

    start_month = 1
    end_month = 12

    if args.year == current_year:
        end_month = current_month
        print(f"Generating data for {args.year} from January to the current month ({datetime.date(1900, current_month, 1).strftime('%B')}).")
    elif args.year == current_year - 1: # Previous year
        print(f"Generating data for the previous year ({args.year}) for all months.")
    else:
        print(f"Error: Data generation is only supported for the current year ({current_year}) or the previous year ({current_year - 1}).")
        exit()

    all_monthly_data = []

    for month in range(start_month, end_month + 1):
        print(f"\n--- Generating monthly data for {datetime.date(1900, month, 1).strftime('%B')} {args.year} ---")
        df_month = generate_data_for_month(args.year, month, args.rows)
        all_monthly_data.append(df_month)
        
    if not all_monthly_data:
        print("No monthly data generated. Exiting.")
        exit()

    full_df_monthly = pd.concat(all_monthly_data, ignore_index=True)
    
    initial_full_rows = len(full_df_monthly)
    full_df_monthly.drop_duplicates(subset=config.unique_columns, inplace=True)
    rows_after_full_dedup = len(full_df_monthly)
    print(f"\nRemoved {initial_full_rows - rows_after_full_dedup} duplicate rows from combined monthly data. Remaining rows: {rows_after_full_dedup}")
    

    all_generated_dfs = [full_df_monthly] # Start with granular data

    # Helper function to generate summaries
    def generate_summaries(df, dimensions_to_group_by, data_type_suffix, static_labels=None):
        summaries = []
        if df.empty:
            return pd.DataFrame()

        load_ts = datetime.datetime.now().isoformat()
        
        # Ensure all dimensions_to_group_by are present in the DataFrame
        for dim in dimensions_to_group_by:
            if dim not in df.columns:
                df[dim] = None # Add missing columns

        for metric_name in df['metric_name'].unique():
            rollup_func = get_rollup_function(metric_name, config.metrics_info)
            df_metric_filtered = df[df['metric_name'] == metric_name].copy()

            if not df_metric_filtered.empty:
                # Group by the specified dimensions and fixed metric columns
                aggregated_df = df_metric_filtered.groupby(
                    dimensions_to_group_by + ["metric_name", "high_or_low_better"], 
                    as_index=False
                ).agg(metric_value=('metric_value', rollup_func))

                # Apply static labels (e.g., "All Products", "All Regions")
                if static_labels:
                    for col, val in static_labels.items():
                        # Ensure column exists before assigning, create if not
                        if col not in aggregated_df.columns:
                            aggregated_df[col] = None 
                        aggregated_df[col] = val
                
                aggregated_df['data_type'] = data_type_suffix
                aggregated_df['load_ts'] = load_ts
                
                # Dynamic summary generation
                # Ensure all columns used in summary_base exist, even if they are None/NaN
                for col in ['product_name', 'service_type', 'region', 'rpt_mth', 'quarter_temp', 'rpt_year']:
                    if col not in aggregated_df.columns:
                        aggregated_df[col] = None

                aggregated_df['summary_base'] = aggregated_df.apply(lambda row: ' '.join(filter(None, [
                    f"Product '{row['product_name']}'" if 'product_name' in row and pd.notna(row['product_name']) and row['product_name'] != "All Products" else "All Products",
                    f"Service '{row['service_type']}'" if 'service_type' in row and pd.notna(row['service_type']) and row['service_type'] != "All Services" else "All Services",
                    f"Region '{row['region']}'" if 'region' in row and pd.notna(row['region']) and row['region'] != "All Regions" else "All Regions",
                    # Conditional time dimension based on data_type_suffix
                    (f"Month '{row['rpt_mth']}'" if 'rpt_mth' in row and pd.notna(row['rpt_mth']) and "Monthly" in data_type_suffix else None),
                    (f"Quarter '{row['rpt_mth']}'" if 'rpt_mth' in row and pd.notna(row['rpt_mth']) and "Quarterly" in data_type_suffix else None),
                    f"Year '{row['rpt_year']}'" if 'rpt_year' in row and pd.notna(row['rpt_year']) else None
                ])), axis=1)

                aggregated_df['summary'] = aggregated_df.apply(
                    lambda row: f"""{row['summary_base'].strip()} has a total metric of {row['metric_name']} with a value of {row['metric_value']:.2f}, where {row['high_or_low_better']} ({data_type_suffix})""",
                    axis=1
                )
                aggregated_df.drop(columns=['summary_base'], inplace=True)
                summaries.append(aggregated_df)
        return pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()

    # Create a temporary df for quarterly aggregation to add 'quarter_temp'
    temp_df_for_quarterly_agg = full_df_monthly.copy()
    if not temp_df_for_quarterly_agg.empty:
        temp_df_for_quarterly_agg['quarter_temp'] = temp_df_for_quarterly_agg['rpt_mth'].apply(
            lambda x: get_quarter_name(datetime.datetime.strptime(x, '%B').month)
        )

    # Get all unique service types across all products for "All Products - Individual Service" drill-down
    all_unique_service_types = set()
    for services_list in config.products.values():
        all_unique_service_types.update(services_list)
    all_unique_service_types = sorted(list(all_unique_service_types)) # Sort for consistent order

    # Define aggregation levels for iteration
    product_levels = list(config.products.keys()) + ["All Products"]
    region_levels = config.regions + ["All Regions"]
    time_levels = ["monthly", "quarterly", "yearly"]

    # Generate all combinations of summaries
    for current_product_level in product_levels:
        # Determine the set of service types to iterate for the current product level
        if current_product_level == "All Products":
            # When product is "All Products", we can summarize by "All Services" OR by individual service types (across all products)
            service_types_to_iterate_for_this_product_level = all_unique_service_types + ["All Services"]
        else:
            # When product is individual, we summarize by its specific services or "All Services" for that product
            service_types_to_iterate_for_this_product_level = config.products.get(current_product_level, []) + ["All Services"]
            service_types_to_iterate_for_this_product_level = sorted(list(set(service_types_to_iterate_for_this_product_level))) # Remove duplicates

        for current_service_level in service_types_to_iterate_for_this_product_level:
            for current_region_level in region_levels:
                for current_time_level in time_levels:

                    grouping_dims = [] # Dimensions to group by for generate_summaries
                    static_labels_for_summary = {} # Labels to apply after aggregation
                    data_type_suffix_parts = []
                    df_base_for_time = full_df_monthly # Base DF for time level, will be updated

                    # --- Determine Time Dimension and base DataFrame ---
                    if current_time_level == 'monthly':
                        grouping_dims.extend(["rpt_mth", "rpt_year"])
                        data_type_suffix_parts.insert(0, "Monthly")
                        df_base_for_time = full_df_monthly
                        static_labels_for_summary['quarter_temp'] = None # quarter_temp is not relevant for monthly
                    elif current_time_level == 'quarterly':
                        grouping_dims.extend(["rpt_mth", "rpt_year"])
                        data_type_suffix_parts.insert(0, "Quarterly")
                        
                        # Create a new DataFrame for quarterly aggregation
                        df_base_for_time = temp_df_for_quarterly_agg.copy()
                        # Overwrite rpt_mth with quarter_temp values
                        df_base_for_time['rpt_mth'] = df_base_for_time['quarter_temp']
                        # Drop the original 'quarter_temp' column as its values are now in 'rpt_mth'
                        df_base_for_time = df_base_for_time.drop(columns=['quarter_temp'])
                        
                        # Remove quarter_temp from static_labels_for_summary if it was added in a previous iteration
                        if 'quarter_temp' in static_labels_for_summary:
                            del static_labels_for_summary['quarter_temp']
                        # Ensure rpt_mth is not set to None or "Full Year" if it was from a previous iteration
                        if 'rpt_mth' in static_labels_for_summary and static_labels_for_summary['rpt_mth'] in [None, "Full Year"]:
                            del static_labels_for_summary['rpt_mth']

                    elif current_time_level == 'yearly':
                        grouping_dims.append("rpt_year")
                        static_labels_for_summary['rpt_mth'] = "Full Year"
                        static_labels_for_summary['quarter_temp'] = None
                        data_type_suffix_parts.insert(0, "Yearly")
                        df_base_for_time = full_df_monthly # For yearly, can use either as only year is relevant

                    # --- Filter the DataFrame based on specific individual levels ---
                    # This pre-filtering is crucial for 'individual' levels to reduce the data before aggregation
                    df_to_aggregate = df_base_for_time.copy()

                    # Product Dimension Filtering/Labeling
                    if current_product_level != "All Products":
                        df_to_aggregate = df_to_aggregate[df_to_aggregate['product_name'] == current_product_level]
                        static_labels_for_summary['product_name'] = current_product_level
                        data_type_suffix_parts.append(current_product_level.replace(" ", "_"))
                    else:
                        static_labels_for_summary['product_name'] = "All Products"
                        data_type_suffix_parts.append("All_Prod")

                    # Service Dimension Filtering/Labeling
                    if current_service_level != "All Services":
                        df_to_aggregate = df_to_aggregate[df_to_aggregate['service_type'] == current_service_level]
                        static_labels_for_summary['service_type'] = current_service_level
                        data_type_suffix_parts.append(current_service_level.replace(" ", "_"))
                    else: # current_service_level is "All Services"
                        static_labels_for_summary['service_type'] = "All Services"
                        data_type_suffix_parts.append("All_Service")

                    # Region Dimension Filtering/Labeling
                    if current_region_level != "All Regions":
                        df_to_aggregate = df_to_aggregate[df_to_aggregate['region'] == current_region_level]
                        static_labels_for_summary['region'] = current_region_level
                        data_type_suffix_parts.append(current_region_level.replace(" ", "_"))
                    else:
                        static_labels_for_summary['region'] = "All Regions"
                        data_type_suffix_parts.append("All_Region")

                    data_type_suffix = "_".join(data_type_suffix_parts) + "_Summary"

                    print(f"\n--- Generating {data_type_suffix.replace('_', ' ')} ---")

                    summary_df = generate_summaries(
                        df_to_aggregate, # Pass the pre-filtered DataFrame
                        dimensions_to_group_by=grouping_dims, # Only time dimensions for grouping within generate_summaries
                        static_labels=static_labels_for_summary,
                        data_type_suffix=data_type_suffix
                    )
                    all_generated_dfs.append(summary_df)


    # Filter out empty dataframes before concatenation
    all_generated_dfs = [df for df in all_generated_dfs if not df.empty]

    if all_generated_dfs:
        final_df = pd.concat(all_generated_dfs, ignore_index=True)
    else:
        final_df = pd.DataFrame(columns=config.unique_columns + ['metric_value', 'summary', 'load_ts', 'data_type', 'high_or_low_better']) # Ensure columns exist even if empty

    # Ensure final_df has unique_columns for final deduplication
    if not final_df.empty:
        # Add missing unique_columns if they somehow got dropped or were not created for some summary
        for col in config.unique_columns:
            if col not in final_df.columns:
                final_df[col] = None 
        
        # Drop quarter_temp if it exists, as it's a temporary column for quarterly aggregation
        if 'quarter_temp' in final_df.columns:
            final_df.drop(columns=['quarter_temp'], inplace=True)

        # Drop duplicates based on the defined unique columns
        initial_final_rows = len(final_df)
        final_df.drop_duplicates(subset=config.unique_columns, inplace=True)
        rows_after_final_dedup = len(final_df)
        print(f"\nRemoved {initial_final_rows - rows_after_final_dedup} duplicate rows from combined final data. Remaining rows: {rows_after_final_dedup}")

        # Reorder columns to a consistent order for output
        desired_columns_order = [
            "product_name", "service_type", "region", "rpt_mth", "rpt_year", 
            "metric_name", "metric_value", "high_or_low_better", "data_type", "load_ts", "summary"
        ]
        # Add any columns that might be in final_df but not in desired_columns_order at the end
        extra_cols = [col for col in final_df.columns if col not in desired_columns_order]
        final_df = final_df[desired_columns_order + extra_cols]


        # Save the final DataFrame to a CSV file in the 'data' folder
        os.makedirs("data", exist_ok=True)
        output_filename = os.path.join("data", f"generated_telecom_data_{args.year}.csv")
        final_df.to_csv(output_filename, index=False)
        print(f"\nGenerated data saved to {output_filename}")

        # Verification of Wireless Net Adds business rules
        wireless_net_adds_df = final_df[(final_df['product_name'] == 'Wireless') & 
                                        (final_df['data_type'] == 'Monthly') &
                                        (final_df['metric_name'].isin(['Wireless Gross Adds', 'Wireless Disconnects', 'Wireless Net Adds']))].copy()
        
        if not wireless_net_adds_df.empty:
            wireless_violations = []
            grouped_wireless = wireless_net_adds_df.groupby(['rpt_mth', 'rpt_year', 'region', 'service_type'])
            
            for name, group in grouped_wireless:
                rpt_mth, rpt_year, region, service_type = name
                gross_adds = group[group['metric_name'] == 'Wireless Gross Adds']['metric_value'].sum()
                disconnects = group[group['metric_name'] == 'Wireless Disconnects']['metric_value'].sum()
                net_adds = group[group['metric_name'] == 'Wireless Net Adds']['metric_value'].sum()

                if abs(net_adds - (gross_adds - disconnects)) > 0.01:
                    wireless_violations.append(
                        f"Violation at {rpt_mth} {rpt_year} {region} {service_type}: "
                        f"Wireless Net Adds ({net_adds:.2f}) != Wireless Gross Adds ({gross_adds:.2f}) - Wireless Disconnects ({disconnects:.2f})"
                    )
            
            if wireless_violations:
                print("\nFound Wireless Net Adds Business Rule Violations:")
                for violation in wireless_violations:
                    print(f"  - {violation}")
            else:
                print("\nWireless Net Adds business rule is satisfied at the monthly level.")
        else:
            print("No Wireless Net Adds monthly data to verify business rules.")

        # Verification for Customer Trouble Tickets Business Rule
        customer_trouble_tickets_parent = "Customer Trouble Tickets Count"
        customer_trouble_tickets_components = [
            "Customer Trouble Tickets - Call Drop",
            "Customer Trouble Tickets - Discount Related",
            "Customer Trouble Tickets - Network Coverage",
            "Customer Trouble Tickets - Slow Net Speed"
        ]

        customer_tickets_df = final_df[(final_df['product_name'] == 'Wireless') &
                                       (final_df['data_type'] == 'Monthly') &
                                       (final_df['metric_name'].isin([customer_trouble_tickets_parent] + customer_trouble_tickets_components))].copy()

        if not customer_tickets_df.empty:
            customer_tickets_violations = []
            grouped_tickets = customer_tickets_df.groupby(['rpt_mth', 'rpt_year', 'region', 'service_type'])

            for name, group in grouped_tickets:
                rpt_mth, rpt_year, region, service_type = name
                
                parent_val = group[group['metric_name'] == customer_trouble_tickets_parent]['metric_value'].sum()
                sum_of_components = 0
                for comp_metric in customer_trouble_tickets_components:
                    sum_of_components += group[group['metric_name'] == comp_metric]['metric_value'].sum()

                if abs(parent_val - sum_of_components) > 0.01: # Allowing a small float tolerance
                    customer_tickets_violations.append(
                        f"Violation at {rpt_mth} {rpt_year} {region} {service_type}: "
                        f"'{customer_trouble_tickets_parent}' ({parent_val:.2f}) != Sum of components ({sum_of_components:.2f})"
                    )

            if customer_tickets_violations:
                print("\nFound Customer Trouble Tickets Business Rule Violations:")
                for violation in customer_tickets_violations:
                    print(f"  - {violation}")
            else:
                print("\nCustomer Trouble Tickets business rule is satisfied at the monthly level.")
        else:
            print("No Wireless Customer Trouble Tickets monthly data to verify business rules.")

        # Verification for Number of Customer Interactions Business Rule
        customer_interactions_parent = "Number of Customer Interactions"
        customer_interactions_components = [
            "Number of Customer Interactions - App",
            "Number of Customer Interactions - Call Centers",
            "Number of Customer Interactions - Portal",
            "Number of Customer Interactions - Store",
            "Number Of Customers with Autopay Discount",
            "Number Of Customers with Late Fee Waiver"
        ]

        customer_interactions_df = final_df[(final_df['product_name'] == 'Wireless') &
                                            (final_df['data_type'] == 'Monthly') &
                                            (final_df['metric_name'].isin([customer_interactions_parent] + customer_interactions_components))].copy()

        if not customer_interactions_df.empty:
            customer_interactions_violations = []
            grouped_interactions = customer_interactions_df.groupby(['rpt_mth', 'rpt_year', 'region', 'service_type'])

            for name, group in grouped_interactions:
                rpt_mth, rpt_year, region, service_type = name
                parent_val_df = group[group['metric_name'] == customer_interactions_parent]['metric_value'].sum()
                
                sum_of_components = 0
                for comp_metric in customer_interactions_components:
                    sum_of_components += group[group['metric_name'] == comp_metric]['metric_value'].sum()

                if abs(parent_val_df - sum_of_components) > 0.01:
                    customer_interactions_violations.append(
                        f"Violation at {rpt_mth} {rpt_year} {region} {service_type}: "
                        f"'{customer_interactions_parent}' (DF: {parent_val_df:.2f}) != Sum of components ({sum_of_components:.2f})"
                    )
            
            if customer_interactions_violations:
                print("\nFound Number of Customer Interactions Business Rule Violations:")
                for violation in customer_interactions_violations:
                    print(f"  - {violation}")
            else:
                print("\nNumber of Customer Interactions business rule is satisfied at the monthly level.")
        else:
            print("No Wireless Customer Interactions monthly data to verify business rules.")
    else:
        print("No monthly data generated to verify rollups and business rules.")
