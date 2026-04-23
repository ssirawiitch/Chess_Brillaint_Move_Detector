# Chess Brilliant Move Predict

## Project Summary
The **Chess Brilliant Move Predictor** is an AI-powered system designed to identify "Brilliant Moves" (annotated as `!!` in chess) using a lightweight Large Language Model (LLM). Instead of relying on traditional, high-latency engine calculations (like Stockfish) during real-time play, this project fine-tunes the **Qwen-2.5-1.5B** model to recognize complex tactical patterns, sacrifices, and evaluation swings.

By utilizing **LoRA (Low-Rank Adaptation)** and a **Hybrid Feature Approach**, the system learns to mimic the intuition of a grandmaster. It classifies whether a specific move meets the "Brilliant" criteria based on a combination of the board's **FEN (Forsyth-Edwards Notation)** and key extracted numerical features (e.g., material difference, engine evaluation delta).

---

## Project Workflow

### 1. Data Extraction & Feature Engineering
* **Parsing PGNs:** Scan professional or high-rated game files (`.pgn`) to find moves explicitly annotated with the `!!` symbol.
* **Feature Collection:** For every move, extract the board state **before** the move (FEN) and compute advanced features such as `delta_eval`, `material_diff` (to explicitly detect sacrifices), and `is_check`.
* **Handling Imbalance (Downsampling):** Since brilliant moves are extremely rare, randomly sample "Normal" moves to roughly match the count of brilliant moves, creating a balanced dataset for the LLM to learn efficiently.

### 2. Data Formatting (JSONL)
* **Optimization:** Convert the processed tabular data into **JSONL (JSON Lines)** format. This ensures memory efficiency, allowing the model to stream the data line-by-line during training without crashing the RAM.
* **Hybrid Schema:** Each line incorporates both the FEN string and the calculated numerical features into a rich, contextual prompt:
  `{"text": "Board: [FEN] | Move: [Move] | Is Check: [True/False] | Material Diff: [Value] | Delta Eval: [Score] | Label: [Brilliant/Normal]"}`

### 3. Fine-tuning with LoRA
* **Base Model:** Load the **Qwen-2.5-1.5B** model in 4-bit quantization to minimize VRAM usage.
* **Adapter Training:** Apply **LoRA** to insert small, trainable matrices into the model's layers. This allows the model to learn chess-specific logic while keeping the original "knowledge" of the base model frozen.
* **Pattern Recognition:** The model learns to associate specific board configurations and numerical contexts (e.g., sacrificing material for a massive evaluation spike) with the "Brilliant" label.

### 4. Inference & Prediction
* **Deployment:** Combine the trained LoRA adapter with the base Qwen model.
* **Prediction:** Input a new board state and its corresponding calculated features. The model outputs a classification (Yes/No) indicating whether the move is considered "Brilliant" based on the patterns it learned during training.