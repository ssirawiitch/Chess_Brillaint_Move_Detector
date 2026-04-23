import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_model():
    base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    adapter_dir = os.path.join(script_dir, '..', 'models', 'chess-lora')
    
    if not os.path.exists(adapter_dir):
        print(f"Error: LoRA adapter not found at {adapter_dir}")
        print("Please run train_lora.py first.")
        return None, None
        
    print("Loading base model...")
    # Load base model (using bfloat16 to match training, saves RAM)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    return model, tokenizer

def predict_move(model, tokenizer, fen, move, player_elo, elo_diff, material_diff, num_legal_moves, is_check, delta_eval):
    is_check_str = "True" if is_check else "False"
    
    if isinstance(delta_eval, float):
        delta_eval_str = f"{delta_eval:.2f}"
    else:
        delta_eval_str = str(delta_eval)
        
    # Format the prompt exactly like the training data, but WITHOUT the Label part at the end
    prompt = (
        f"Board: {fen} | "
        f"Move: {move} | "
        f"Player Elo: {player_elo} | "
        f"Elo Diff: {elo_diff} | "
        f"Material Diff: {material_diff} | "
        f"Legal Moves: {num_legal_moves} | "
        f"Is Check: {is_check_str} | "
        f"Delta Eval: {delta_eval_str} | "
        f"Label:"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate prediction
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5, # We only need it to say "Brilliant" or "Normal"
            temperature=0.1,  # Low temperature = more confident, less random
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
    # Decode the newly generated part only
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    prediction = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    return prediction

if __name__ == "__main__":
    model, tokenizer = load_model()
    
    if model is not None:
        print("\n--- Testing Chess Brilliant Move Predictor ---")
        
        # Test Case 1: A situation that looks like a Brilliant sacrifice
        print("\nTest Case 1: Queen Sacrifice for checkmate advantage")
        pred1 = predict_move(
            model=model,
            tokenizer=tokenizer,
            fen="r1b1k2r/pppp1ppp/8/4P3/1bB4q/2N3P1/PPP2P1P/R1BQK2R b KQkq - 0 9", # Mock FEN
            move="Qxf2+", 
            player_elo=2200,
            elo_diff=50,
            material_diff=-9, # Sacrificing the Queen!
            num_legal_moves=35,
            is_check=True,
            delta_eval=8.50 # Engine evaluates this as completely winning
        )
        print(f"Model Prediction: {pred1} (Expected: Brilliant)")
        
        # Test Case 2: A very standard normal opening move
        print("\nTest Case 2: Normal opening pawn push")
        pred2 = predict_move(
            model=model,
            tokenizer=tokenizer,
            fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            move="e4", 
            player_elo=1500,
            elo_diff=0,
            material_diff=0, 
            num_legal_moves=20,
            is_check=False,
            delta_eval=0.30
        )
        print(f"Model Prediction: {pred2} (Expected: Normal)")
