import os
import json
import torch
import argparse
import sys
from transformers import RobertaTokenizerFast, RobertaForTokenClassification
from peft import PeftModel

# Add parent directory of 'src' to path to handle package imports when run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def print_model_config(model_dir):
    """
    Read adapter_config.json from model_dir and print PEFT/LoRA configuration.
    """
    config_path = os.path.join(model_dir, "adapter_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            print("--------------------------------------------------")
            print(f"[PEFT/LoRA] Loaded Model Config from '{model_dir}':")
            print(f"  - Base Model:     {config.get('base_model_name_or_path')}")
            print(f"  - LoRA Rank (r):  {config.get('r')}")
            print(f"  - LoRA Alpha (a): {config.get('lora_alpha')}")
            print(f"  - Target Modules: {', '.join(config.get('target_modules', []))}")
            print("--------------------------------------------------")
        except Exception as e:
            print(f"Warning: Could not parse adapter_config.json: {e}")
    else:
        print(f"Warning: adapter_config.json not found in '{model_dir}'.")

def post_process_bio_tags(tags):
    """
    Correct invalid BIO tag sequences.
    For example:
    - Change B-XYZ to I-XYZ if it immediately follows B-XYZ or I-XYZ of the same type.
    - Change I-XYZ to B-XYZ if it starts a sequence (e.g. after O).
    """
    corrected_tags = []
    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            ent_type = tag.split("-")[1]
            if i > 0:
                prev_tag = corrected_tags[i-1]
                if prev_tag.endswith(ent_type) and (prev_tag.startswith("B-") or prev_tag.startswith("I-")):
                    corrected_tags.append(f"I-{ent_type}")
                    continue
        elif tag.startswith("I-"):
            ent_type = tag.split("-")[1]
            if i == 0:
                corrected_tags.append(f"B-{ent_type}")
                continue
            else:
                prev_tag = corrected_tags[i-1]
                if not (prev_tag.endswith(ent_type) and (prev_tag.startswith("B-") or prev_tag.startswith("I-"))):
                    corrected_tags.append(f"B-{ent_type}")
                    continue
        corrected_tags.append(tag)
    return corrected_tags

def predict_entities(text, model_dir="results/best_model", mappings_path="results/tag_mappings.json"):
    """
    Load the best trained LoRA model and predict NER tags for a given raw text.
    Runs 100% pure machine learning inference using the model weights.
    """
    # 1. Load Tag Mappings
    if not os.path.exists(mappings_path):
        print(f"Error: Tag mappings not found at '{mappings_path}'. Please run training first.")
        return
        
    with open(mappings_path, "r") as f:
        mappings = json.load(f)
    id2tag = {int(k): v for k, v in mappings["id2tag"].items()}
    num_labels = len(id2tag)
    
    # 2. Check if best model exists
    if not os.path.exists(model_dir):
        print(f"Error: Best model directory not found at '{model_dir}'. Please run training/grid search first.")
        return
        
    print(f"Loading best LoRA model from '{model_dir}'...")
    print_model_config(model_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Running inference on: '{device}'")
    
    # Load tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", add_prefix_space=True)
    
    # Check if adapter_config.json exists (LoRA vs non-LoRA)
    adapter_config_path = os.path.join(model_dir, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        print(f"[Loading Mode] PEFT/LoRA adapter detected in '{model_dir}'")
        base_model = RobertaForTokenClassification.from_pretrained("roberta-base", num_labels=num_labels)
        model = PeftModel.from_pretrained(base_model, model_dir)
    else:
        print(f"[Loading Mode] Standard Full Fine-Tuning weights detected in '{model_dir}'")
        model = RobertaForTokenClassification.from_pretrained(model_dir, num_labels=num_labels)
        
    model.to(device)
    model.eval()
    
    # 3. Tokenize input text
    words = text.split()
    inputs = tokenizer(
        words, 
        is_split_into_words=True, 
        return_tensors="pt", 
        truncation=True
    )
    
    # Move inputs to device (retaining BatchEncoding class to keep word_ids method)
    inputs = inputs.to(device)
    
    # 4. Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2)[0].tolist()
        
    # 5. Align predictions back to words
    word_ids = inputs.word_ids(batch_index=0)
    previous_word_idx = None
    predicted_tags = []
    
    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        if word_idx != previous_word_idx:
            pred_id = predictions[token_idx]
            predicted_tags.append(id2tag.get(pred_id, "O"))
        previous_word_idx = word_idx
        
    # Zip words with their predicted tags
    results = list(zip(words, predicted_tags))
    
    # Apply BIO transition rule post-processing
    corrected_tags = post_process_bio_tags(predicted_tags)
    corrected_results = list(zip(words, corrected_tags))
    
    print("\n--- Hasil Prediksi Pelabelan NER (Raw) ---")
    for word, tag in results:
        print(f"{word:20} -> {tag}")
        
    print("\n--- Hasil Prediksi Pelabelan NER (Setelah Koreksi Aturan BIO) ---")
    for word, tag in corrected_results:
        print(f"{word:20} -> {tag}")
            
    return corrected_results

def predict_dataset(data_path, output_path="results/whole_dataset_predictions.json", model_dir="results/best_model", mappings_path="results/tag_mappings.json"):
    """
    Load a full dataset from data_path, predict NER tags for each sample using the best model,
    and save the combined tokens, true tags, and predicted tags to output_path.
    Runs 100% pure machine learning inference using the model weights.
    """
    from src.data_loader import load_data
    
    # Load data
    data = load_data(data_path)
    if not data:
        print(f"Error: No data loaded from '{data_path}'")
        return
        
    # Load Tag Mappings
    if not os.path.exists(mappings_path):
        print(f"Error: Tag mappings not found at '{mappings_path}'")
        return
    with open(mappings_path, "r") as f:
        mappings = json.load(f)
    id2tag = {int(k): v for k, v in mappings["id2tag"].items()}
    num_labels = len(id2tag)
    
    # Load model
    if not os.path.exists(model_dir):
        print(f"Error: Best model directory not found at '{model_dir}'")
        return
        
    print(f"Loading best model and running inference on {len(data)} samples...")
    print_model_config(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Running inference on: '{device}'")
    
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", add_prefix_space=True)
    
    # Check if adapter_config.json exists (LoRA vs non-LoRA)
    adapter_config_path = os.path.join(model_dir, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        print(f"[Loading Mode] PEFT/LoRA adapter detected in '{model_dir}'")
        base_model = RobertaForTokenClassification.from_pretrained("roberta-base", num_labels=num_labels)
        model = PeftModel.from_pretrained(base_model, model_dir)
    else:
        print(f"[Loading Mode] Standard Full Fine-Tuning weights detected in '{model_dir}'")
        model = RobertaForTokenClassification.from_pretrained(model_dir, num_labels=num_labels)
        
    model.to(device)
    model.eval()
    
    results = []
    
    # Initialize hidden entities tracking (for report purposes only)
    hidden_keywords = ["monster", "creature", "wretch", "fiend", "demon", "creator"]
    hidden_stats = {k: {"total": 0, "detected": 0} for k in hidden_keywords}
    
    # Batch/loop prediction
    for idx, sample in enumerate(data):
        words = sample.get("tokens", [])
        true_tags = sample.get("tags", [])
        if not words:
            continue
            
        inputs = tokenizer(
            words, 
            is_split_into_words=True, 
            return_tensors="pt", 
            truncation=True
        )
        
        # Move inputs to device (retaining BatchEncoding class to keep word_ids method)
        inputs = inputs.to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=2)[0].tolist()
            
        word_ids = inputs.word_ids(batch_index=0)
        previous_word_idx = None
        predicted_tags = []
        
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            if word_idx != previous_word_idx:
                pred_id = predictions[token_idx]
                predicted_tags.append(id2tag.get(pred_id, "O"))
            previous_word_idx = word_idx
            
        # Handle cases where truncation happened in tokenizer
        if len(predicted_tags) < len(words):
            predicted_tags += ["O"] * (len(words) - len(predicted_tags))
        elif len(predicted_tags) > len(words):
            predicted_tags = predicted_tags[:len(words)]
            
        # Track hidden entity detection rates
        for w, true_t, pred_t in zip(words, true_tags, predicted_tags):
            clean_word = w.lower().strip(".,;:!?\"'()[]{}")
            if clean_word in hidden_keywords:
                hidden_stats[clean_word]["total"] += 1
                if "PERSON" in pred_t:
                    hidden_stats[clean_word]["detected"] += 1
            
        results.append({
            "tokens": words,
            "true_tags": true_tags,
            "predicted_tags": predicted_tags
        })
        
        if (idx + 1) % 100 == 0:
            print(f"Predicted {idx + 1}/{len(data)} samples...")
            
    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
    print(f"\nSuccess! Saved whole dataset predictions to: '{output_path}'")
    
    # Compile and display Hidden Entity Detection Report
    print("\n==================================================")
    print("      LITERARY HIDDEN ENTITY DETECTION REPORT      ")
    print("==================================================")
    total_all = 0
    detected_all = 0
    for keyword, stats in hidden_stats.items():
        total = stats["total"]
        detected = stats["detected"]
        total_all += total
        detected_all += detected
        acc = (detected / total * 100.0) if total > 0 else 0.0
        print(f"- '{keyword:10}': Detected {detected}/{total} times (Recall: {acc:.2f}%)")
    print("--------------------------------------------------")
    overall_acc = (detected_all / total_all * 100.0) if total_all > 0 else 0.0
    print(f"Overall Hidden Entity Recall: {detected_all}/{total_all} ({overall_acc:.2f}%)")
    print("==================================================")
    
    # Save the report as text file for easy copy-paste to thesis
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    report_path = os.path.join(os.path.dirname(output_path), f"{base_name}_hidden_entity_report.txt")
    with open(report_path, "w", encoding="utf-8") as rf:
        rf.write("==================================================\n")
        rf.write("      LITERARY HIDDEN ENTITY DETECTION REPORT      \n")
        rf.write("==================================================\n")
        for keyword, stats in hidden_stats.items():
            total = stats["total"]
            detected = stats["detected"]
            acc = (detected / total * 100.0) if total > 0 else 0.0
            rf.write(f"- '{keyword:10}': Detected {detected}/{total} times (Recall: {acc:.2f}%)\n")
        rf.write("--------------------------------------------------\n")
        rf.write(f"Overall Hidden Entity Recall: {detected_all}/{total_all} ({overall_acc:.2f}%)\n")
        rf.write("==================================================\n")
    print(f"Detailed analysis report saved to: '{report_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script to label raw text or whole dataset using the best trained model.")
    parser.add_argument("--text", type=str, default=None, help="Single raw text sentence to predict")
    parser.add_argument("--data_path", type=str, default=None, help="Path to input dataset file (e.g. data/frankenstein_annotated.json) to predict on")
    parser.add_argument("--output_path", type=str, default="results/whole_dataset_predictions.json", help="Path to save predictions when predicting on a dataset")
    parser.add_argument("--model_dir", type=str, default="results/best_model", help="Path to best model directory")
    
    args = parser.parse_args()
    
    if args.data_path:
        predict_dataset(args.data_path, args.output_path, args.model_dir)
    elif args.text:
        predict_entities(args.text, args.model_dir)
    else:
        # If no arguments provided, run on default single text
        default_text = "I saw Victor Frankenstein and the Monster in Geneva."
        predict_entities(default_text, args.model_dir)
