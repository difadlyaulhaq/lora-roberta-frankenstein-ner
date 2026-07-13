import os
import json
import torch
import argparse
import sys
from transformers import RobertaTokenizerFast, RobertaForTokenClassification
from peft import PeftModel

# Add parent directory of 'src' to path to handle package imports when run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def predict_entities(text, model_dir="results/best_model", mappings_path="results/tag_mappings.json"):
    """
    Load the best trained LoRA model and predict NER tags for a given raw text.
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
    
    # Load tokenizer and base model
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", add_prefix_space=True)
    base_model = RobertaForTokenClassification.from_pretrained("roberta-base", num_labels=num_labels)
    
    # Load LoRA adapter weights on top of the base model
    model = PeftModel.from_pretrained(base_model, model_dir)
    model.eval()
    
    # 3. Tokenize input text
    words = text.split()
    inputs = tokenizer(
        words, 
        is_split_into_words=True, 
        return_tensors="pt", 
        truncation=True
    )
    
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
    
    print("\n--- Hasil Prediksi Pelabelan NER ---")
    for word, tag in results:
        if tag != "O":
            print(f"{word:20} -> {tag}")
        else:
            print(f"{word:20} -> O")
            
    return results

def predict_dataset(data_path, output_path="results/whole_dataset_predictions.json", model_dir="results/best_model", mappings_path="results/tag_mappings.json"):
    """
    Load a full dataset from data_path, predict NER tags for each sample using the best model,
    and save the combined tokens, true tags, and predicted tags to output_path.
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
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", add_prefix_space=True)
    base_model = RobertaForTokenClassification.from_pretrained("roberta-base", num_labels=num_labels)
    model = PeftModel.from_pretrained(base_model, model_dir)
    model.eval()
    
    results = []
    
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
