from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import torch, json
from huggingface_hub import login, hf_hub_download
import os
import logging
import shutil # Import shutil for file operations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Make sure you have your HUGGINGFACE_API_TOKEN in config.py
try:
    from config import HUGGINGFACE_API_TOKEN
    login(token=HUGGINGFACE_API_TOKEN)
except ImportError:
    logger.warning("config.py not found or HUGGINGFACE_API_TOKEN not set. Skipping Hugging Face login.")
    HUGGINGFACE_API_TOKEN = None
except Exception as e:
    logger.error(f"Error during Hugging Face login: {e}")
    HUGGINGFACE_API_TOKEN = None

samples = []
data_file_path = "logs/distillation_data.jsonl"
if not os.path.exists(data_file_path):
    logger.error(f"Error: Data file not found at {data_file_path}. Please ensure it exists and is accessible.")
    exit()

try:
    with open(data_file_path, "r") as f:
        for line in f:
            item = json.loads(line)
            if 'query' in item and 'data' in item and 'teacher_output' in item:
                input_text = f"User Query: {item['query']}\nData: {json.dumps(item['data'])}"
                output_text = item['teacher_output']
                samples.append({"input": input_text, "output": output_text})
            else:
                logger.warning(f"Skipping malformed line in JSONL: {line.strip()} (missing 'query', 'data', or 'teacher_output')")
except json.JSONDecodeError as e:
    logger.error(f"Error decoding JSON from {data_file_path}: {e}")
    exit()
except Exception as e:
    logger.error(f"An unexpected error occurred while reading {data_file_path}: {e}")
    exit()

if not samples:
    logger.error("Error: No samples loaded from the dataset. Training cannot proceed.")
    exit()
else:
    logger.info(f"Loaded {len(samples)} samples from {data_file_path}")

dataset = Dataset.from_list(samples)

model_name = "google/gemma-2-2b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Tokenizer pad_token was None, set to eos_token: {tokenizer.pad_token_id}")

MAX_LENGTH = 512

def tokenize(example):
    prompt = f"### Question:\n{example['input']}\n\n### Answer:\n{example['output']}{tokenizer.eos_token}"

    full_tokenized = tokenizer(
        prompt,
        truncation=True,
        max_length=MAX_LENGTH,
        # IMPORTANT: Do NOT use return_tensors here. DataCollatorForLanguageModeling handles this.
        # return_tensors="pt"
    )

    question_prefix = f"### Question:\n{example['input']}\n\n### Answer:"
    # Use encode(add_special_tokens=False) to get token IDs without bos/eos for prefix length
    # Note: If `question_prefix` itself exceeds MAX_LENGTH after tokenization, this calculation
    # might be inaccurate relative to the truncated `full_tokenized` sequence.
    # For simplicity and to address the current error, we ensure list output.
    question_tokenized_ids = tokenizer.encode(question_prefix, add_special_tokens=False, truncation=True, max_length=MAX_LENGTH)
    question_tokenized_len = len(question_tokenized_ids)


    # Ensure input_ids and attention_mask are plain Python lists
    input_ids = full_tokenized["input_ids"]
    attention_mask = full_tokenized["attention_mask"]

    # Create a new list for labels and apply masking
    labels = list(input_ids) # Create a copy as a plain Python list
    for i in range(min(question_tokenized_len, len(labels))):
        labels[i] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

logger.info("Tokenizing dataset...")
tokenized = dataset.map(tokenize, remove_columns=dataset.column_names, batched=False, load_from_cache_file=False)

logger.info("Dataset tokenization complete. Checking first tokenized example:")
if len(tokenized) > 0:
    first_example = tokenized[0]
    logger.info(f"First example input_ids length: {len(first_example['input_ids'])}")
    logger.info(f"First example labels (first 20): {first_example['labels'][:20]}")
    logger.info(f"First example attention_mask (first 20): {first_example['attention_mask'][:20]}")
    if -100 in first_example['labels']:
        logger.info("Labels contain -100, masking seems to be applied.")
    else:
        logger.warning("Labels do not contain -100. Masking of prompt might not be working as expected.")

if torch.cuda.is_available():
    logger.info("GPU detected, loading model with torch_dtype=torch.bfloat16 and device_map='auto'.")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation='eager')
else:
    logger.warning("No GPU detected, loading model on CPU. This will be significantly slower.")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu", torch_dtype=torch.float32, attn_implementation='eager')
    # If running on CPU and bitsandbytes warning appears, it's expected as bitsandbytes is for GPU quantization/optimization.

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, config)
model.print_trainable_parameters()

lora_output_dir = "lora_output"
os.makedirs(lora_output_dir, exist_ok=True)

args = TrainingArguments(
    output_dir=lora_output_dir,
    per_device_train_batch_size=2, # Reduced batch size as requested
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    fp16=False, # Set to True if using GPU with fp16 support
    report_to="none",
    logging_first_step=True,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
)

if not tokenized or len(tokenized) == 0:
    logger.error("Tokenized dataset is empty. Cannot start training.")
    exit()

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    # Explicitly add padding=True if it helps, though it's usually default behavior
    # This ensures all sequences in a batch are padded to the longest sequence in that batch or max_length
    padding="longest", # or "max_length"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    tokenizer=tokenizer, # Keep tokenizer here for backward compatibility warnings
    data_collator=data_collator, # Explicitly pass the data collator
)

logger.info("Starting training...")
try:
    trainer.train()
    logger.info("Training complete.")
except Exception as e:
    logger.error(f"An error occurred during training: {e}", exc_info=True)

# --- Save LoRA adapters only ---
lora_adapters_path = "student_model_lora_adapters"
os.makedirs(lora_adapters_path, exist_ok=True)
model.save_pretrained(lora_adapters_path)
tokenizer.save_pretrained(lora_adapters_path)
logger.info(f"\nLoRA training complete and adapters saved to '{lora_adapters_path}'.")

# --- SECTION FOR OLLAMA INTEGRATION AND MERGING ---
# This section requires significant RAM (32GB+ recommended for Gemma 2B)
# and possibly GPU for faster merging/saving.
# If you are on a 16GB RAM CPU laptop, this will likely fail due to memory.
# It's best to run this part on a more powerful machine or cloud instance.

merged_model_output_dir = "student_model_merged"
os.makedirs(merged_model_output_dir, exist_ok=True)

logger.info(f"Loading base model {model_name} for merging LoRA adapters...")
# Load the base model with the same dtype and device_map as used for training
if torch.cuda.is_available():
    base_model_full = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
else:
    base_model_full = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu", torch_dtype=torch.float32)

logger.info(f"Loading LoRA adapters from '{lora_adapters_path}'...")
peft_model = PeftModel.from_pretrained(base_model_full, lora_adapters_path)

logger.info("Merging LoRA adapters into the base model (memory intensive)...")
try:
    # Ensure all components are on the same device before merging
    if torch.cuda.is_available():
        peft_model.to("cuda") # Move peft_model to GPU for merging
    merged_model = peft_model.merge_and_unload()
    logger.info("LoRA adapters merged successfully.")

    logger.info(f"Attempting to save merged model to '{merged_model_output_dir}' (disk & memory intensive)...")
    merged_model.save_pretrained(merged_model_output_dir, safe_serialization=True)
    
    # --- CRITICAL ADDITION: Copy tokenizer.model from base model's cache ---
    # This ensures llama.cpp finds the expected tokenizer.model file.
    # Use hf_hub_download to get the path to the original tokenizer.model file
    original_tokenizer_model_path = hf_hub_download(repo_id=model_name, filename="tokenizer.model")
    
    if original_tokenizer_model_path and os.path.exists(original_tokenizer_model_path):
        destination_tokenizer_model = os.path.join(merged_model_output_dir, "tokenizer.model")
        shutil.copy(original_tokenizer_model_path, destination_tokenizer_model)
        logger.info(f"Copied original tokenizer.model to '{destination_tokenizer_model}'.")
    else:
        logger.warning(f"tokenizer.model not found from original model cache for {model_name}. Attempting to save current tokenizer...")
        # Fallback: Save the current tokenizer (might still save tokenizer.json only if that's its internal representation)
        tokenizer.save_pretrained(merged_model_output_dir)
        if not os.path.exists(os.path.join(merged_model_output_dir, "tokenizer.model")):
             logger.error("Still unable to find or create tokenizer.model in merged output. `llama.cpp` conversion might still fail.")
             logger.error("You might need to manually find `tokenizer.model` from the original `google/gemma-2-2b` HF repo and place it in the merged model directory.")


    logger.info(f"Merged model and tokenizer successfully saved to '{merged_model_output_dir}'.")

except torch.cuda.OutOfMemoryError:
    logger.error("--- FAILED TO MERGE/SAVE MODEL: CUDA OUT OF MEMORY ---", exc_info=True)
    logger.error("Your GPU likely does not have enough VRAM to complete the merge and save operation.")
    logger.error("Consider using a larger GPU or performing this step on a CPU with more RAM (e.g., 32GB+).")
    logger.error("If on CPU, close other applications to free up RAM.")
except Exception as e:
    logger.error(f"--- FAILED TO MERGE/SAVE MERGED MODEL DUE TO RESOURCE OR OTHER ISSUES ---", exc_info=True)
    logger.error("This error ('Insufficient system resources' or other) means your system lacks sufficient **DISK SPACE** or **RAM** to complete the save operation.")
    logger.error("1. **Check Disk Space:** Ensure you have enough free space on the drive where you are saving.")
    logger.error("2. **Monitor RAM Usage:** During model loading, merging, and saving, this process can temporarily consume significant RAM (10+ GB for Gemma 2B on CPU). Close other applications.")
    logger.error(f"Save directory was: {os.path.abspath(merged_model_output_dir)}")


logger.info("Training script finished (attempted).")