import os
import numpy as np
import pandas as pd
import torch
from transformers import Trainer, TrainingArguments, DataCollatorForTokenClassification, AutoTokenizer
from seqeval.metrics import f1_score, precision_score, recall_score
from .model import get_roberta_lora_model
from .evaluate import profile_computing_performance

def get_compute_metrics_fn(id2tag):
    """
    Generate compute_metrics function mapped with specific class id2tag dictionary.
    """
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index -100 (padding and subwords)
        true_predictions = [
            [id2tag[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2tag[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions)
        }
    return compute_metrics

def train_model(model, train_dataset, eval_dataset, training_args, id2tag, data_collator=None):
    """
    Main training loop utilizing Hugging Face Trainer.
    """
    compute_metrics_fn = get_compute_metrics_fn(id2tag)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn
    )
    trainer.train()
    return trainer

def grid_search_lora(train_dataset, eval_dataset, id2tag, model_name="roberta-base", num_labels=19, output_dir="./results"):
    """
    Perform grid search over different values of LoRA Rank (r) and Alpha (alpha).
    """
    r_values = [4, 8, 16]
    alpha_values = [8, 16, 32]
    
    results = []
    
    for r in r_values:
        for alpha in alpha_values:
            print(f"\n========================================")
            print(f"Training LoRA model with rank={r}, alpha={alpha}")
            print(f"========================================")
            
            # Setup directories
            exp_dir = os.path.join(output_dir, f"lora_r{r}_a{alpha}")
            os.makedirs(exp_dir, exist_ok=True)
            
            # Define training arguments
            training_args = TrainingArguments(
                output_dir=exp_dir,
                eval_strategy="epoch",
                save_strategy="epoch",
                learning_rate=5e-4,  # Standard learning rate for LoRA tuning
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                num_train_epochs=3,
                weight_decay=0.01,
                logging_steps=10,
                load_best_model_at_end=True,
                metric_for_best_model="f1",
                report_to="none"
            )
            
            # Prepare tokenizer for data collator
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            data_collator = DataCollatorForTokenClassification(tokenizer)
            
            def init_and_train():
                # Re-initialize the model to isolate experiments
                model = get_roberta_lora_model(
                    model_name=model_name,
                    num_labels=num_labels,
                    r=r,
                    lora_alpha=alpha
                )
                trainer = train_model(
                    model=model,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    training_args=training_args,
                    id2tag=id2tag,
                    data_collator=data_collator
                )
                return trainer, model

            # Train with performance profiling (time & VRAM)
            (trainer, model), training_time, peak_vram = profile_computing_performance(init_and_train)
            
            print(f"Finished training in {training_time:.2f} seconds. Peak VRAM: {peak_vram:.4f} GB")
            
            # Evaluate model performance
            eval_metrics = trainer.evaluate()
            f1 = eval_metrics.get("eval_f1", 0.0)
            precision = eval_metrics.get("eval_precision", 0.0)
            recall = eval_metrics.get("eval_recall", 0.0)
            
            print(f"Eval results: F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
            
            results.append({
                "rank": r,
                "alpha": alpha,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "training_time_sec": training_time,
                "peak_vram_gb": peak_vram
            })
            
            # Clean up memory to avoid accumulation across runs
            if torch.cuda.is_available():
                del model
                del trainer
                torch.cuda.empty_cache()
                
    # Save search logs
    df_results = pd.DataFrame(results)
    results_path = os.path.join(output_dir, "grid_search_results.csv")
    df_results.to_csv(results_path, index=False)
    print(f"\nGrid search completed! Results saved to: {results_path}")
    
    return df_results

