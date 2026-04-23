import pandas as pd
import json
import os

def prepare_data():
    csv_path = '../data/raw_features.csv'
    train_jsonl_path = '../data/train.jsonl'
    test_jsonl_path = '../data/test.jsonl'
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, '..', 'data', 'raw_features.csv')
    train_jsonl_path = os.path.join(script_dir, '..', 'data', 'train.jsonl')
    test_jsonl_path = os.path.join(script_dir, '..', 'data', 'test.jsonl')
    
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        print("Please ensure you have run extract_features.py first.")
        return

    print("Loading data...")
    df = pd.read_csv(csv_path)
    
    # Handle missing delta_eval if any
    df['delta_eval'] = df['delta_eval'].fillna(0.0)
    
    # Phase 2: Step 3 - Downsampling
    print(f"Total rows before downsampling: {len(df)}")
    
    # Separate brilliant and normal moves
    brilliant_df = df[df['is_brilliant'] == 1]
    normal_df = df[df['is_brilliant'] == 0]
    
    num_brilliant = len(brilliant_df)
    print(f"Found {num_brilliant} brilliant moves.")
    
    if num_brilliant == 0:
        print("Warning: No brilliant moves found in the dataset! Cannot balance. Exiting.")
        return
        
    # Sample normal moves to be 2 times the number of brilliant moves (1:2 ratio)
    target_normal = num_brilliant * 2
    if len(normal_df) > target_normal:
        normal_sampled = normal_df.sample(n=target_normal, random_state=42)
    else:
        normal_sampled = normal_df
        
    # Combine and shuffle
    balanced_df = pd.concat([brilliant_df, normal_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Balanced dataset size: {len(balanced_df)} rows (1:2 ratio)")
    
    # Split 80/20 for Train and Test
    train_df = balanced_df.sample(frac=0.8, random_state=42)
    test_df = balanced_df.drop(train_df.index)
    
    print(f"Train size: {len(train_df)} rows")
    print(f"Test size: {len(test_df)} rows")
    
    # Phase 2: Step 4 - Convert to JSONL
    def save_to_jsonl(dataframe, path):
        print(f"Saving to {path}...")
        with open(path, 'w', encoding='utf-8') as f:
            for _, row in dataframe.iterrows():
                label_str = "Brilliant" if row['is_brilliant'] == 1 else "Normal"
                
                # Using the Hybrid Schema + new features we added
                is_check_bool = "True" if row['is_check'] == 1 else "False"
                
                # Format delta eval nicely
                delta_eval = row['delta_eval']
                if isinstance(delta_eval, float):
                    delta_eval_str = f"{delta_eval:.2f}"
                else:
                    delta_eval_str = str(delta_eval)
                
                text_prompt = (
                    f"Board: {row['fen']} | "
                    f"Move: {row['move']} | "
                    f"Player Elo: {row['player_elo']} | "
                    f"Elo Diff: {row['elo_diff']} | "
                    f"Material Diff: {row['material_diff']} | "
                    f"Legal Moves: {row['num_legal_moves']} | "
                    f"Is Check: {is_check_bool} | "
                    f"Delta Eval: {delta_eval_str} | "
                    f"Label: {label_str}"
                )
                
                json_line = {"text": text_prompt}
                f.write(json.dumps(json_line) + '\n')
                
    save_to_jsonl(train_df, train_jsonl_path)
    save_to_jsonl(test_df, test_jsonl_path)
            
    print("Done! Data is ready for fine-tuning and evaluation.")

if __name__ == '__main__':
    prepare_data()
