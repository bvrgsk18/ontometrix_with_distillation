# Import necessary libraries
from transformers import AutoTokenizer, TrainingArguments, Gemma3ForCausalLM
from trl import SFTTrainer
from peft import LoraConfig
from datasets import load_dataset
from huggingface_hub import login
#from google.colab import userdata
import torch
from config import *
# Ensure your JSONL file is correctly formatted for the SFTTrainer.
# It should contain a 'text' field with the data to be used for fine-tuning.
dataset = load_dataset('json', data_files='logs/metrics_dataset_gemma3.jsonl', split='train')

# Authenticate with Hugging Face Hub
HF_TOKEN = HUGGINGFACE_API_TOKEN#userdata.get('HF_TOKEN')
login(token=HF_TOKEN)

# Define the model name
model_name = "google/gemma-3-4b-it"



# Load the tokenizer and adjust padding settings
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" 

# Explicitly set the device to CPU since no GPU is available
device = torch.device("cpu")
print(f"Using device: {device}")

# Load the Gemma 3 model. We are not using a device map or specific attention
# implementation since we are on a CPU.
model = Gemma3ForCausalLM.from_pretrained(
    model_name,attn_implementation='eager'
)
model.config.use_cache = False  # Disable caching for training

# Move the model to the CPU
model.to(device)

# Set up LoRA configuration for causal language modeling

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

# Define training arguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    learning_rate=2e-4,
    logging_steps=1,
    save_steps=25,
    report_to="tensorboard",
    group_by_length=True,
)

# Create the SFTTrainer with LoRA parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    args=training_args,
    max_seq_length=1024,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("gemma3_finetuned")
