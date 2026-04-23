import os
import pathlib

# Fix for Windows Thai locale (cp874) error when importing trl
_original_read_text = pathlib.Path.read_text
def _patched_read_text(self, encoding=None, errors=None):
    if encoding is None:
        encoding = 'utf-8'
    return _original_read_text(self, encoding=encoding, errors=errors)
pathlib.Path.read_text = _patched_read_text

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

def train():
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, '..', 'data', 'train.jsonl')
    eval_dataset_path = os.path.join(script_dir, '..', 'data', 'test.jsonl')
    output_dir = os.path.join(script_dir, '..', 'models', 'chess-lora')
    
    # 1. Load dataset
    print("Loading datasets...")
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found. Please run prepare_jsonl.py first.")
        return
        
    dataset = load_dataset('json', data_files={'train': dataset_path, 'test': eval_dataset_path})
    
    # 2. Configure BitsAndBytes for 4-bit quantization (Saves VRAM)
    print("Loading model in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 # Change to torch.float16 if GPU doesn't support bfloat16
    )
    
    # 3. Load Base Model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model.config.use_cache = False
    
    # 4. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # 5. LoRA Config (r=16, alpha=32)
    print("Setting up LoRA...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 6. Training Arguments (using SFTConfig for newer trl versions)
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=5, # Train for 5 epochs (can adjust based on dataset size)
        save_strategy="epoch",
        eval_strategy="epoch",
        optim="paged_adamw_32bit",
        fp16=False,
        bf16=True, # Disable if GPU doesn't support bfloat16 (e.g. older than RTX 3000)
        max_grad_norm=0.3,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        report_to="none",
        dataset_text_field="text",
        max_length=256
    )
    
    # 7. SFTTrainer
    print("Initializing Trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'], # Evaluating on our test.jsonl!
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
    )
    
    # 8. Start Training
    print("Starting training...")
    trainer.train()
    
    # 9. Save final model
    print(f"Saving final adapter to {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done! You now have a fine-tuned LoRA model.")

if __name__ == "__main__":
    train()
