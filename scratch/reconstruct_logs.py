import os
import json
import re

results_dir = "results"
output_file = os.path.join(results_dir, "full_training_log.txt")

# Standard run directories
standard_dirs = ["standard_run_no_lora", "standard_run"]
# Find all lora grid search directories
lora_dirs = [d for d in os.listdir(results_dir) if d.startswith("lora_") and os.path.isdir(os.path.join(results_dir, d))]

# Sort grid search directories for a nice ordered log (r4 -> r8 -> r16 -> r32)
def parse_lora_name(name):
    match = re.match(r"lora_r(\d+)_a(\d+)", name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 999, 999

lora_dirs.sort(key=parse_lora_name)

# Combine all directories to parse in order
all_dirs_to_parse = []
for sd in standard_dirs:
    if os.path.exists(os.path.join(results_dir, sd)):
        all_dirs_to_parse.append((sd, sd.replace("_", " ").upper()))
        
for ld in lora_dirs:
    r, a = parse_lora_name(ld)
    all_dirs_to_parse.append((ld, f"LoRA Grid Search (rank={r}, alpha={a})"))

print(f"Found {len(all_dirs_to_parse)} training directories to process. Reconstructing logs...")

with open(output_file, "w", encoding="utf-8") as out:
    out.write("========================================================================\n")
    out.write("RECONSTRUCTED COMPLETE TRAINING LOGS FROM ALL EXPERIMENTS\n")
    out.write("========================================================================\n\n")

    for dir_name, display_title in all_dirs_to_parse:
        dir_path = os.path.join(results_dir, dir_name)
        
        # Find all checkpoints
        checkpoints = [c for c in os.listdir(dir_path) if c.startswith("checkpoint-") and os.path.isdir(os.path.join(dir_path, c))]
        if not checkpoints:
            print(f"Warning: No checkpoints found in {dir_name}")
            continue
            
        # Get checkpoint with highest number
        checkpoints.sort(key=lambda x: int(x.split("-")[1]))
        latest_checkpoint = checkpoints[-1]
        
        state_file = os.path.join(dir_path, latest_checkpoint, "trainer_state.json")
        if not os.path.exists(state_file):
            print(f"Warning: trainer_state.json not found in {os.path.join(dir_name, latest_checkpoint)}")
            continue
            
        print(f"Processing {dir_name} using {latest_checkpoint}...")
        
        with open(state_file, "r") as f:
            state_data = json.load(f)
            
        out.write(f"========================================================================\n")
        out.write(f"Training Config: {display_title}\n")
        out.write(f"========================================================================\n")
        out.write(f"Best Global Step: {state_data.get('best_global_step')}\n")
        out.write(f"Best Metric (eval_f1): {state_data.get('best_metric'):.6f}\n")
        out.write(f"Total Epochs: {state_data.get('epoch')}\n")
        out.write(f"Total Steps: {state_data.get('global_step')}\n\n")
        
        log_history = state_data.get("log_history", [])
        for log in log_history:
            if "loss" in log:
                # This is a training step log
                step = log.get("step")
                epoch = log.get("epoch")
                loss = log.get("loss")
                lr = log.get("learning_rate")
                grad_norm = log.get("grad_norm", "N/A")
                out.write(f"{{'loss': {loss:.4f}, 'grad_norm': {grad_norm}, 'learning_rate': {lr}, 'epoch': {epoch:.2f}, 'step': {step}}}\n")
            elif "eval_loss" in log:
                # This is an evaluation step log
                epoch = log.get("epoch")
                eval_loss = log.get("eval_loss")
                eval_p = log.get("eval_precision", 0.0)
                eval_r = log.get("eval_recall", 0.0)
                eval_f1 = log.get("eval_f1", 0.0)
                out.write(f"\n--- Evaluation at Epoch {epoch:.2f} ---\n")
                out.write(f"{{'eval_loss': {eval_loss:.6f}, 'eval_precision': {eval_p:.6f}, 'eval_recall': {eval_r:.6f}, 'eval_f1': {eval_f1:.6f}}}\n\n")
                
        out.write("\n\n")

print(f"\nSuccess! Reconstructed training logs written to: {output_file}")
