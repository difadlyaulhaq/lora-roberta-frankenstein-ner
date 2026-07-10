import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizerFast

class NERDataset(Dataset):
    """
    A PyTorch Dataset wrapper for tokenized NER data.
    """
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

def load_data(file_path):
    """
    Load BIO data from a JSON or JSONL file.
    Expected format is a list of objects or line-by-line objects:
    {"tokens": ["word1", "word2"], "tags": ["O", "B-PERSON"]}
    """
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content.startswith("["):
                # Standard JSON array
                data = json.loads(content)
            else:
                # JSON Lines (JSONL)
                for line in content.splitlines():
                    if line.strip():
                        data.append(json.loads(line))
        print(f"Successfully loaded {len(data)} examples from {file_path}")
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        # Return an empty list or try to load as CSV if possible
        if file_path.endswith(".csv"):
            try:
                df = pd.read_csv(file_path)
                for _, row in df.iterrows():
                    # Check if tokens and tags are string-serialized lists or comma-separated
                    tokens = eval(row["tokens"]) if isinstance(row["tokens"], str) and row["tokens"].startswith("[") else str(row["tokens"]).split()
                    tags = eval(row["tags"]) if isinstance(row["tags"], str) and row["tags"].startswith("[") else str(row["tags"]).split()
                    data.append({"tokens": tokens, "tags": tags})
                print(f"Successfully loaded {len(data)} examples from CSV: {file_path}")
            except Exception as csv_err:
                print(f"Failed loading as CSV: {csv_err}")
    return data

def get_tag_mappings(data):
    """
    Extract unique tags from data and create mappings to/from integer IDs.
    Ensures 'O' is mapped to index 0.
    """
    unique_tags = set()
    for item in data:
        unique_tags.update(item["tags"])
    
    sorted_tags = sorted(list(unique_tags))
    if "O" in sorted_tags:
        sorted_tags.remove("O")
        sorted_tags = ["O"] + sorted_tags
        
    tag2id = {tag: i for i, tag in enumerate(sorted_tags)}
    id2tag = {i: tag for i, tag in enumerate(sorted_tags)}
    return tag2id, id2tag

def preprocess_and_tokenize(data, tokenizer: RobertaTokenizerFast, tag2id):
    """
    Preprocess raw text and tokenize into token-level inputs aligned with BIO labels.
    """
    texts = [x["tokens"] for x in data]
    tags = [x["tags"] for x in data]
    
    tokenized_inputs = tokenizer(
        texts,
        is_split_into_words=True,
        truncation=True,
        padding=True
    )
    
    labels = []
    for i, label in enumerate(tags):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                # Set special tokens (like padding or start/end tokens) to -100
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # Only label the first token of a given word
                label_ids.append(tag2id.get(label[word_idx], tag2id.get("O", 0)))
            else:
                # Set the other subtokens of a word to -100
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
        
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

