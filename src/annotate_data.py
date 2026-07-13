import os
import json
import re
from transformers import pipeline

def sent_tokenize(text):
    """
    A simple sentence tokenizer using regular expressions.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def word_tokenize(sentence):
    """
    A simple word tokenizer.
    """
    # Keep alphanumeric characters, punctuation separated
    tokens = re.findall(r'\w+|[^\w\s]', sentence, re.UNICODE)
    return tokens

def annotate_text(raw_json_path, output_json_path, model_name="tner/bert-base-ontonotes5"):
    """
    Load raw chapter data, tokenize into sentences, detect entities using a pre-trained 
    OntoNotes NER pipeline, and save in BIO tag format.
    """
    print(f"Loading raw text from {raw_json_path}...")
    if not os.path.exists(raw_json_path):
        print(f"Error: Raw text JSON not found at {raw_json_path}")
        return

    with open(raw_json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # 1. Initialize Hugging Face NER Pipeline
    print(f"Loading pre-trained NER model: {model_name}...")
    try:
        # aggregation_strategy="simple" groups subword predictions into word-level entities
        ner_pipeline = pipeline("ner", model=model_name, aggregation_strategy="simple")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to a standard model (dslim/bert-base-NER)...")
        ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

    annotated_dataset = []
    
    # Process each chapter
    for chapter in raw_data.get("chapters", []):
        chapter_title = chapter.get("title", "")
        chapter_content = chapter.get("content", "")
        print(f"Processing chapter: {chapter_title}...")
        
        # Tokenize into sentences
        sentences = sent_tokenize(chapter_content)
        
        # Run NER pipeline on sentences in batches
        for sent in sentences:
            if not sent:
                continue
                
            # Get word tokens
            tokens = word_tokenize(sent)
            if not tokens:
                continue
                
            # Initialize all tags as "O" (Outside)
            tags = ["O"] * len(tokens)
            
            # Predict entities in the sentence
            try:
                entities = ner_pipeline(sent)
            except Exception as e:
                # Handle potential truncation/errors gracefully
                entities = []
                
            # Align predicted entities with our word tokens
            # HF simple aggregation yields: [{'entity_group': 'PERSON', 'word': 'Victor', 'start': 0, 'end': 6}]
            for ent in entities:
                ent_type = ent.get("entity_group")
                ent_word = ent.get("word", "").strip()
                start_char = ent.get("start")
                end_char = ent.get("end")
                
                # Find matching token indices
                current_char_idx = 0
                first_token_idx = None
                last_token_idx = None
                
                for t_idx, token in enumerate(tokens):
                    # Find token start/end positions in sentence
                    match_pos = sent.find(token, current_char_idx)
                    if match_pos == -1:
                        continue
                    
                    tok_start = match_pos
                    tok_end = match_pos + len(token)
                    current_char_idx = tok_end
                    
                    # Check overlap with entity char range
                    if tok_start >= start_char and tok_end <= end_char:
                        if first_token_idx is None:
                            first_token_idx = t_idx
                        last_token_idx = t_idx
                
                # Apply BIO labels to matched tokens
                if first_token_idx is not None and last_token_idx is not None:
                    tags[first_token_idx] = f"B-{ent_type}"
                    for t_idx in range(first_token_idx + 1, last_token_idx + 1):
                        tags[t_idx] = f"I-{ent_type}"
            
            annotated_dataset.append({
                "tokens": tokens,
                "tags": tags
            })
            
    # Save annotated data
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(annotated_dataset, f, indent=4, ensure_ascii=False)
        
    print(f"\nSuccess! Generated {len(annotated_dataset)} annotated samples.")
    print(f"Annotated dataset saved to: {output_json_path}")

if __name__ == "__main__":
    raw_json = "data/frankenstein_data.json"
    output_json = "data/frankenstein_annotated_auto.json"
    annotate_text(raw_json, output_json)
