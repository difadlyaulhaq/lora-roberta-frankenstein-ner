import os
import sys
import json
import argparse
import time
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizerFast, TrainingArguments, DataCollatorForTokenClassification

# Add parent directory of 'src' to path to handle package imports when run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_data, get_tag_mappings, preprocess_and_tokenize, NERDataset
from src.model import get_roberta_lora_model, get_roberta_baseline_model
from src.train import train_model, grid_search_lora, get_compute_metrics_fn
from src.evaluate import evaluate_linguistic_metrics, evaluate_muc5_errors, evaluate_fine_grained, evaluate_hidden_entities

def main():
    parser = argparse.ArgumentParser(description="LoRA-RoBERTa Token Classification Pipeline")
    parser.add_argument("--data_path", type=str, default="data/frankenstein_annotated.json", help="Path to BIO annotated JSON/JSONL dataset")
    parser.add_argument("--model_name", type=str, default="roberta-base", help="Pretrained RoBERTa model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save models and logs")
    parser.add_argument("--grid_search", action="store_true", help="Run hyperparameter grid search over LoRA rank/alpha")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training/evaluation")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate for parameters")
    parser.add_argument("--no_lora", action="store_true", help="Train baseline model using standard Full Fine-Tuning (no LoRA adapter)")
    
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
                    
                    # Automatically run whole dataset inference on the best model
                    print("\nRunning automatic whole dataset prediction on the Best Grid Search model...")
                    from src.predict import predict_dataset
                    predict_dataset(
                        data_path=args.data_path,
                        output_path=os.path.join(args.output_dir, "grid_search_best_predictions.json"),
                        model_dir=dest_best_dir,
                        mappings_path=os.path.join(args.output_dir, "tag_mappings.json")
                    )
    else:
        # Standard Single Run Training
        if args.no_lora:
            print(f"\nInitializing baseline model with Full Fine-Tuning (no LoRA)")
            model = get_roberta_baseline_model(
                model_name=args.model_name,
                num_labels=num_labels
            )
            # Default learning rate for Full Fine-Tuning is typically smaller (e.g. 5e-5)
            # than LoRA (5e-4) to prevent catastrophic forgetting.
            lr = args.learning_rate if args.learning_rate != 5e-4 else 5e-5
            output_subdir = "standard_run_no_lora"
        else:
            print(f"\nInitializing standard LoRA model (r=8, alpha=16)")
            model = get_roberta_lora_model(
                model_name=args.model_name,
                num_labels=num_labels,
                r=8,
                lora_alpha=16
            )
            lr = args.learning_rate
            output_subdir = "standard_run"
        
        training_args = TrainingArguments(
            output_dir=os.path.join(args.output_dir, output_subdir),
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=lr,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            weight_decay=0.01,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            report_to="none"
        )
        
        # Reset peak VRAM tracker
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        start_time = time.time()
        print("Training model...")
        trainer = train_model(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_args=training_args,
            id2tag=id2tag,
            data_collator=data_collator
        )
        training_time = time.time() - start_time
        
        # Record peak VRAM
        peak_vram = 0.0
        if torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
        
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
            
        # Calculate seqeval metrics for summary CSV
        from seqeval.metrics import f1_score as seqeval_f1, precision_score as seqeval_p, recall_score as seqeval_r
        f1_metric = seqeval_f1(true_labels, true_preds)
        precision_metric = seqeval_p(true_labels, true_preds)
        recall_metric = seqeval_r(true_labels, true_preds)

        # Calculate hidden entity recall
        hidden_results = evaluate_hidden_entities(true_preds, true_labels, sentences=eval_sentences)

        # Save evaluation reports
        eval_report = {
            "overall_f1": f1_metric,
            "classification_report": report,
            "muc5_errors": muc5_results,
            "fine_grained_analysis": fine_grained_results,
            "hidden_entity_analysis": hidden_results
        }
        
        report_filename = "evaluation_report_no_lora.json" if args.no_lora else "evaluation_report.json"
        eval_path = os.path.join(args.output_dir, report_filename)
        with open(eval_path, "w") as f:
            json.dump(eval_report, f, indent=4)
        print(f"\nEvaluation metrics successfully compiled and saved to '{eval_path}'")

        # Compile flat CSV results
        summary_data = {
            "rank": "N/A" if args.no_lora else 8,
            "alpha": "N/A" if args.no_lora else 16,
            "f1": f1_metric,
            "precision": precision_metric,
            "recall": recall_metric,
            "training_time_sec": training_time,
            "peak_vram_gb": peak_vram,
            "muc5_COR": muc5_results.get("COR", 0),
            "muc5_INC": muc5_results.get("INC", 0),
            "muc5_MIS": muc5_results.get("MIS", 0),
            "muc5_SPU": muc5_results.get("SPU", 0),
            "eLen_short_acc": fine_grained_results.get("eLen_short", {}).get("accuracy", 0.0),
            "eLen_long_acc": fine_grained_results.get("eLen_long", {}).get("accuracy", 0.0),
            "eCon_consistent_acc": fine_grained_results.get("eCon_consistent", {}).get("accuracy", 0.0),
            "eCon_inconsistent_acc": fine_grained_results.get("eCon_inconsistent", {}).get("accuracy", 0.0),
            "eFre_few_shot_acc": fine_grained_results.get("eFre_few_shot", {}).get("accuracy", 0.0),
            "eFre_many_shot_acc": fine_grained_results.get("eFre_many_shot", {}).get("accuracy", 0.0),
            "hidden_entity_recall": hidden_results.get("overall_recall", 0.0),
        }
        
        csv_filename = "standard_no_lora_results.csv" if args.no_lora else "standard_lora_results.csv"
        csv_path = os.path.join(args.output_dir, csv_filename)
        df_summary = pd.DataFrame([summary_data])
        df_summary.to_csv(csv_path, index=False)
        print(f"Summary results successfully saved to CSV: {csv_path}")

        # Automatically run whole dataset predictions on the trained standard model
        print(f"\nRunning automatic whole dataset prediction on the trained standard model...")
        from src.predict import predict_dataset
        
        output_pred_filename = "baseline_no_lora_predictions.json" if args.no_lora else "regular_lora_predictions.json"
        # Find the best checkpoint saved in training_args.output_dir
        checkpoints = [c for c in os.listdir(training_args.output_dir) if c.startswith("checkpoint-")]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            model_dir = os.path.join(training_args.output_dir, checkpoints[-1])
        else:
            model_dir = training_args.output_dir
            
        predict_dataset(
            data_path=args.data_path,
            output_path=os.path.join(args.output_dir, output_pred_filename),
            model_dir=model_dir,
            mappings_path=os.path.join(args.output_dir, "tag_mappings.json")
        )

if __name__ == "__main__":
    main()
