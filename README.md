# ontometrix_with_distillation
Ontometrix with Knowledge Distillation.

**Ontometrix** An Ontology-Driven Conversational Agent for Business Metrics Correlation Using Knowledge Graphs

## Project Structure

```
ontometrix_with_distillation/
│
├── app.py                       # Main application entry point
├── build_knowledge_graph.py     # Script to build the knowledge graph
├── config.py                    # Configuration settings
├── data_generator.py            # Data generation utilities
├── distill_logger.py            # Logging for distillation process
├── evaluate_relationships_gold.py # Evaluation script for relationships
├── generate_relationships.py    # Relationship generation logic
├── train_merge.py               # Training script for merging models
├── train_student.py             # Training script for student model
├── Modelfile                    # Model configuration or checkpoint
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
│
├── config/                      # JSON configuration files
│   ├── metrics.json
│   ├── products.json
│   ├── regions.json
│   ├── relationships.json
│   └── unique_columns.json
│
├── data/                        # Datasets and evaluation results
│   ├── generated_telecom_data_2024.csv
│   ├── generated_telecom_data_2025.csv
│   ├── generated_telecom_data_all.csv
│   ├── telecom_metric_relationships.csv
│
├── logs/                        # Log files for experiments
│   ├── distillation_data.jsonl
│   └── metrics_dataset_gemma3.jsonl
│

```

- **app.py**: Main application script.
- **config/**: Configuration files for metrics, products, regions, relationships, and unique columns.
- **data/**: Contains datasets, generated data, and evaluation results.
- **logs/**: Stores logs from distillation and metric dataset experiments.
- **output/**: Contains output files such as the generated ontology.
```


## How to Run

### **Set up environment**
```bash
conda create --name ontometrix python=3.12 -y
conda activate ontometrix
pip install -r requirements.txt
```

### **Execution steps**
```bash
cd ontometrix_with_distillation (project root folder)
conda activate ontometrix

Update config.py with Huggingface token, Gemini API Token, Neo4J instance credentials etc
```

### To start Chatbot Interface
```bash
streamlit run app.py

#### Oprn the localhost url to see the chatbot interface

```

### Data Pipeline

#### **To generate augmented data**
```bash
python data_generator.py --year YYYY 

YYYY is the year for which you want to generate data.
rows is optional parameter if passed it will generate augmented data for NNN rows
```

#### **To detect and generate relationships on augmented data**
```bash
python generate_relationships.py  [-- test]

Supports optional --test mode to run on a limited dataset for which we have ground truth data set curated manually for evalution purpose.
```

#### **To build / modify Knowledge graph based on relationships.**
```bash
python build_knowledge_graph.py

```

#### **To build / modify Knowledge graph based on relationships.**
```bash
python build_knowledge_graph.py

```

### Training Pipeline

#### **for Knowledge distillation /fine tuning the model.**
```bash
python train_student.py

```

#### **To merge lora adapters with base model.**
```bash
python train_merge.py

```

#### **To add student_model to ollamalocal server**
```bash
ollama create student_model -f Modelfile

```

## Author

**B V R G S Kumar**  
M.Tech (AIML), BITS Pilani  

**Project Guide**: Arun Kumar Shanmugam

**Project Evaluator**: Mithun Kumar S R



