import json
import os

def patch_dataset(file_path):
    """
    Read the annotated dataset and correct the labels of hidden entities 
    (monster, creature, wretch, fiend, creator) from 'O' to 'B-PERSON' / 'I-PERSON'
    to fix the training data bias.
    """
    print(f"Reading dataset from {file_path}...")
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return
        
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    hidden_keywords = ["monster", "creature", "wretch", "fiend", "demon", "creator"]
    patched_count = 0
    
    for sentence in data:
        tokens = sentence["tokens"]
        tags = sentence["tags"]
        
        previous_tag = "O"
        for i in range(len(tokens)):
            word = tokens[i]
            clean_word = word.lower().strip(".,;:!?\"'()[]{}")
            
            if clean_word in hidden_keywords:
                # If it was labeled as O, correct it to PERSON
                if tags[i] == "O":
                    if previous_tag.endswith("PERSON"):
                        tags[i] = "I-PERSON"
                    else:
                        tags[i] = "B-PERSON"
                    patched_count += 1
            previous_tag = tags[i]
            
    # Save back to dataset file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        
    print(f"Successfully patched {patched_count} hidden entity labels in {file_path}!")

if __name__ == "__main__":
    patch_dataset("data/frankenstein_annotated.json")
