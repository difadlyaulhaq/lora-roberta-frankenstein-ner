import os
import sys
import json
import argparse
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizerFast, TrainingArguments, DataCollatorForTokenClassification

# Add parent directory of 'src' to path to handle package imports when run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_data, get_tag_mappings, preprocess_and_tokenize, NERDataset
from src.model import get_roberta_lora_model
from src.train import train_model, grid_search_lora, get_compute_metrics_fn
from src.evaluate import evaluate_linguistic_metrics, evaluate_muc5_errors, evaluate_fine_grained

def main():
    parser = argparse.ArgumentParser(description="LoRA-RoBERTa Token Classification Pipeline")
    parser.add_argument("--data_path", type=str, default="data/frankenstein_annotated.json", help="Path to BIO annotated JSON/JSONL dataset")
    parser.add_argument("--model_name", type=str, default="roberta-base", help="Pretrained RoBERTa model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save models and logs")
    parser.add_argument("--grid_search", action="store_true", help="Run hyperparameter grid search over LoRA rank/alpha")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training/evaluation")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate for LoRA parameters")
    
    args = parser.parse_args()
    
    print("--------------------------------------------------")
    print("Initializing LoRA-RoBERTa NER Pipeline")
    print("--------------------------------------------------")
    
    # 1. Load Data
    if not os.path.exists(args.data_path):
        print(f"Error: Annotated dataset file not found at '{args.data_path}'")
        print("Please place your annotated JSON/JSONL/CSV dataset at that path or specify it using --data_path.")
        return
        
    data = load_data(args.data_path)
    if not data:
        print("Error: Loaded dataset is empty.")
        return
        
    # 2. Extract Tag Map
    tag2id, id2tag = get_tag_mappings(data)
    num_labels = len(tag2id)
    print(f"Detected {num_labels} unique tags: {list(tag2id.keys())}")
    
    # Save tag mappings for reference
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "tag_mappings.json"), "w") as f:
        json.dump({"tag2id": tag2id, "id2tag": id2tag}, f, indent=4)
        
    # 3. Train-Test Split (80% Train, 20% Evaluation)
    train_data, eval_data = train_test_split(data, test_size=0.2, random_state=42)
    print(f"Split dataset: {len(train_data)} train samples, {len(eval_data)} evaluation samples.")
    
    # 4. Tokenization and Preprocessing
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = RobertaTokenizerFast.from_pretrained(args.model_name, add_prefix_space=True)
    
    train_encodings = preprocess_and_tokenize(train_data, tokenizer, tag2id)
    eval_encodings = preprocess_and_tokenize(eval_data, tokenizer, tag2id)
    
    train_dataset = NERDataset(train_encodings)
    eval_dataset = NERDataset(eval_encodings)
    
    # Data collator for dynamic token classification padding
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    # 5. Run Execution Mode
    if args.grid_search:
        print("\nStarting LoRA Grid Search Sweep...")
        df_results = grid_search_lora(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            id2tag=id2tag,
            model_name=args.model_name,
            num_labels=num_labels,
            output_dir=args.output_dir,
            eval_data=eval_data,
            epochs=args.epochs
        )
        
        # Determine and finalize the best model
        if df_results is not None and not df_results.empty:
            import shutil
            best_idx = df_results["f1"].idxmax()
            best_row = df_results.loc[best_idx]
            best_r = int(best_row["rank"])
            best_alpha = int(best_row["alpha"])
            best_f1 = best_row["f1"]
            
            print("\n========================================")
            print("GRID SEARCH COMPLETED - BEST MODEL SUMMARY")
            print("========================================")
            print(f"Best Configuration: Rank (r) = {best_r}, Alpha (alpha) = {best_alpha}")
            print(f"Best Validation F1-Score: {best_f1:.6f}")
            print(f"Precision: {best_row['precision']:.6f} | Recall: {best_row['recall']:.6f}")
            print(f"Training Time: {best_row['training_time_sec']:.2f} seconds")
            
            # Find and copy checkpoints for the best model to a dedicated folder
            best_exp_dir = os.path.join(args.output_dir, f"lora_r{best_r}_a{best_alpha}")
            dest_best_dir = os.path.join(args.output_dir, "best_model")
            
            if os.path.exists(best_exp_dir):
                checkpoints = [c for c in os.listdir(best_exp_dir) if c.startswith("checkpoint-")]
                if checkpoints:
                    # Sort numerically to find the highest checkpoint
                    checkpoints.sort(key=lambda x: int(x.split("-")[1]))
                    best_checkpoint_dir = os.path.join(best_exp_dir, checkpoints[-1])
                    
                    if os.path.exists(dest_best_dir):
                        shutil.rmtree(dest_best_dir)
                    
                    shutil.copytree(best_checkpoint_dir, dest_best_dir)
                    print(f"Successfully copied the best model weights to: '{dest_best_dir}'")
    else:
        # Standard Single Run Training
        print(f"\nInitializing standard LoRA model (r=8, alpha=16)")
        model = get_roberta_lora_model(
            model_name=args.model_name,
            num_labels=num_labels,
            r=8,
            lora_alpha=16
        )
        
        training_args = TrainingArguments(
            output_dir=os.path.join(args.output_dir, "standard_run"),
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            weight_decay=0.01,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            report_to="none"
        )
        
        print("Training model...")
        trainer = train_model(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_args=training_args,
            id2tag=id2tag,
            data_collator=data_collator
        )
        
        # 6. Evaluation Suite
        print("\n========================================")
        print("Running Evaluation Suite")
        print("========================================")
        
        # Get raw predictions
        predictions_output = trainer.predict(eval_dataset)
        preds = predictions_output.predictions
        labels = predictions_output.label_ids
        
        # Convert predictions to argmax token classes
        preds_argmax = preds.argmax(axis=2)
        
        # Align predictions and labels back to tag lists, excluding -100
        true_preds = []
        true_labels = []
        for pred, label in zip(preds_argmax, labels):
            true_preds.append([id2tag[p] for p, l in zip(pred, label) if l != -100])
            true_labels.append([id2tag[l] for p, l in zip(pred, label) if l != -100])
            
        # Get evaluation words/sentences for fine-grained analysis
        eval_sentences = [x["tokens"] for x in eval_data]
        
        # Metric 1: Linguistic Metrics
        report, f1 = evaluate_linguistic_metrics(true_preds, true_labels)
        print("\n--- Classification Report ---")
        print(report)
        print(f"Overall F1 Score: {f1:.4f}")
        
        # Metric 2: MUC-5 Errors
        muc5_results = evaluate_muc5_errors(true_preds, true_labels)
        print("\n--- MUC-5 Error Classification ---")
        for metric, val in muc5_results.items():
            print(f"- {metric}: {val}")
            
        # Metric 3: Fine-Grained Analysis
        fine_grained_results = evaluate_fine_grained(true_preds, true_labels, sentences=eval_sentences)
        print("\n--- Fine-Grained Entity Analysis ---")
        for cat, stats in fine_grained_results.items():
            print(f"- {cat:20} -> Total: {stats['total']:<4} | Correct: {stats['correct']:<4} | Accuracy/Recall: {stats['accuracy']:.4f}")
            
        # Save evaluation reports
        eval_report = {
            "overall_f1": f1,
            "classification_report": report,
            "muc5_errors": muc5_results,
            "fine_grained_analysis": fine_grained_results
        }
        
        eval_path = os.path.join(args.output_dir, "evaluation_report.json")
        with open(eval_path, "w") as f:
            json.dump(eval_report, f, indent=4)
        print(f"\nEvaluation metrics successfully compiled and saved to '{eval_path}'")

if __name__ == "__main__":
    main()
