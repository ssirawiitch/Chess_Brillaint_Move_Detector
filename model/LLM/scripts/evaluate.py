import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

def load_model():
    base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    adapter_dir = os.path.join(script_dir, '..', 'models', 'chess-lora')
    
    if not os.path.exists(adapter_dir):
        print(f"Error: LoRA adapter not found at {adapter_dir}")
        return None, None
        
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    # Configure padding for batched inference
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    return model, tokenizer

def evaluate():
    model, tokenizer = load_model()
    if model is None:
        return
        
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_file = os.path.join(script_dir, '..', 'data', 'test.jsonl')
    
    if not os.path.exists(test_file):
        print(f"Error: Test file not found at {test_file}")
        return
        
    with open(test_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    batch_size = 64
    y_true = []
    y_pred = []
    
    print(f"\nStarting evaluation on {len(lines)} samples...")
    print(f"Using batch size: {batch_size}")
    
    # Process in batches
    for i in tqdm(range(0, len(lines), batch_size)):
        batch_lines = lines[i:i+batch_size]
        prompts = []
        labels = []
        
        for line in batch_lines:
            try:
                data = json.loads(line)
                text = data["text"]
                parts = text.split(" | Label: ")
                if len(parts) != 2:
                    continue
                    
                prompt = parts[0] + " | Label:"
                label = parts[1].strip()
                
                prompts.append(prompt)
                labels.append(label)
            except Exception as e:
                continue
                
        if not prompts:
            continue
            
        y_true.extend([1 if l == "Brilliant" else 0 for l in labels])
        
        # Tokenize batch
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        input_length = inputs.input_ids.shape[1]
        
        # Generate predictions
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
        # Decode predictions
        for idx, output in enumerate(outputs):
            generated_tokens = output[input_length:]
            prediction = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            # Map prediction text to binary class
            if "Brilliant" in prediction:
                y_pred.append(1)
            else:
                y_pred.append(0)

    # Calculate Metrics
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n" + "="*50)
    print("EVALUATION METRICS RESULTS")
    print("="*50)
    print(f"Total Test Samples: {total}")
    print(f"True Positives (Brilliant correctly predicted): {tp}")
    print(f"True Negatives (Normal correctly predicted):    {tn}")
    print(f"False Positives (Normal predicted as Brilliant): {fp}")
    print(f"False Negatives (Brilliant predicted as Normal): {fn}")
    print("-" * 50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("="*50)

if __name__ == "__main__":
    evaluate()
