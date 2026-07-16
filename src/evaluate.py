import time
import torch
from seqeval.metrics import classification_report, f1_score
from seqeval.metrics.sequence_labeling import get_entities

def evaluate_linguistic_metrics(predictions, references):
    """
    Calculate Precision, Recall, and F1-Score (Macro-average).
    """
    report = classification_report(references, predictions)
    f1 = f1_score(references, predictions)
    return report, f1

def profile_computing_performance(model_fn, *args, **kwargs):
    """
    Measure peak GPU memory usage (VRAM) in GB and execution time in seconds.
    """
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
    start_time = time.time()
    result = model_fn(*args, **kwargs)
    execution_time = time.time() - start_time
    
    peak_vram = 0.0
    if use_cuda:
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3) # Convert to GB
        
    return result, execution_time, peak_vram

def evaluate_muc5_errors(predictions, references):
    """
    Adapt MUC-5 evaluation guidelines under exact boundary match:
    - COR (Correct): boundary and type match.
    - INC (Incorrect): boundary matches but type differs.
    - MIS (Missing): present in reference but no matching boundary in predictions.
    - SPU (Spurious): predicted but no matching boundary in reference.
    """
    cor, inc, mis, spu = 0, 0, 0, 0
    
    for pred_seq, ref_seq in zip(predictions, references):
        ref_ents = get_entities(ref_seq)
        pred_ents = get_entities(pred_seq)
        
        ref_by_boundary = {(start, end): ent_type for ent_type, start, end in ref_ents}
        pred_by_boundary = {(start, end): ent_type for ent_type, start, end in pred_ents}
        
        # Process reference entities
        for boundary, ref_type in ref_by_boundary.items():
            if boundary in pred_by_boundary:
                pred_type = pred_by_boundary[boundary]
                if ref_type == pred_type:
                    cor += 1
                else:
                    inc += 1
            else:
                mis += 1
                
        # Process spurious predictions
        for boundary in pred_by_boundary:
            if boundary not in ref_by_boundary:
                spu += 1
                
    return {
        "COR": cor,
        "INC": inc,
        "MIS": mis,
        "SPU": spu
    }

def evaluate_fine_grained(predictions, references, sentences=None):
    """
    Fine-grained analysis based on entity properties:
    - eLen (Entity Length >= 4 words vs Short < 4 words)
    - eCon (Label Consistency: Consistent vs Inconsistent entity text)
    - eFre (Entity Frequency: Few-shot/Zero-shot <= 2 vs Many-shot > 2)
    """
    from collections import Counter
    
    # 1. Extract all reference entities and their texts if sentences are available
    ref_entities_list = [] # List of tuples: (sent_idx, start, end, type, text)
    entity_text_labels = {} # Maps entity_text -> list of types
    entity_text_freq = Counter()
    
    for sent_idx, ref_seq in enumerate(references):
        ref_ents = get_entities(ref_seq)
        for ent_type, start, end in ref_ents:
            if sentences and sent_idx < len(sentences):
                text = " ".join(sentences[sent_idx][start:end]).lower()
            else:
                text = f"entity_{start}_{end}"
            
            ref_entities_list.append((sent_idx, start, end, ent_type, text))
            entity_text_freq[text] += 1
            if text not in entity_text_labels:
                entity_text_labels[text] = []
            entity_text_labels[text].append(ent_type)
            
    # Calculate consistency for each text
    # Consistency = frequency of most common label / total frequency
    entity_consistency = {}
    for text, labels in entity_text_labels.items():
        most_common_count = Counter(labels).most_common(1)[0][1]
        entity_consistency[text] = most_common_count / len(labels)
        
    # 2. Track predictions
    pred_entities_by_sent = []
    for pred_seq in predictions:
        pred_ents = get_entities(pred_seq)
        pred_entities_by_sent.append({(start, end): ent_type for ent_type, start, end in pred_ents})
        
    # 3. Categorize and evaluate
    categories = {
        "eLen_short": {"total": 0, "correct": 0},
        "eLen_long": {"total": 0, "correct": 0},
        "eCon_consistent": {"total": 0, "correct": 0},
        "eCon_inconsistent": {"total": 0, "correct": 0},
        "eFre_few_shot": {"total": 0, "correct": 0},
        "eFre_many_shot": {"total": 0, "correct": 0}
    }
    
    for sent_idx, start, end, ref_type, text in ref_entities_list:
        # Check if correctly predicted
        correct = False
        if sent_idx < len(pred_entities_by_sent):
            pred_sent = pred_entities_by_sent[sent_idx]
            if (start, end) in pred_sent and pred_sent[(start, end)] == ref_type:
                correct = True
                
        # Length category
        length = end - start
        len_cat = "eLen_long" if length >= 4 else "eLen_short"
        categories[len_cat]["total"] += 1
        if correct:
            categories[len_cat]["correct"] += 1
            
        # Consistency category
        con = entity_consistency[text]
        con_cat = "eCon_consistent" if con == 1.0 else "eCon_inconsistent"
        categories[con_cat]["total"] += 1
        if correct:
            categories[con_cat]["correct"] += 1
            
        # Frequency category
        freq = entity_text_freq[text]
        fre_cat = "eFre_few_shot" if freq <= 2 else "eFre_many_shot"
        categories[fre_cat]["total"] += 1
        if correct:
            categories[fre_cat]["correct"] += 1
            
    # Calculate recall accuracy for each category
    results = {}
    for cat, stats in categories.items():
        total = stats["total"]
        correct = stats["correct"]
        acc = correct / total if total > 0 else 0.0
        results[cat] = {
            "total": total,
            "correct": correct,
            "accuracy": acc
        }
        
    return results

def evaluate_hidden_entities(predictions, references, sentences):
    """
    Calculate detection recall for hidden entities (monster, creature, wretch, fiend, demon, creator)
    on the evaluation set.
    """
    hidden_keywords = ["monster", "creature", "wretch", "fiend", "demon", "creator"]
    stats = {k: {"total": 0, "detected": 0} for k in hidden_keywords}
    
    for pred_seq, ref_seq, words in zip(predictions, references, sentences):
        # Align predictions back to words
        for w, true_t, pred_t in zip(words, ref_seq, pred_seq):
            clean_word = w.lower().strip(".,;:!?\"'()[]{}")
            if clean_word in hidden_keywords:
                stats[clean_word]["total"] += 1
                if "PERSON" in pred_t:
                    stats[clean_word]["detected"] += 1
                    
    # Calculate overall recall
    total_all = sum(stats[k]["total"] for k in hidden_keywords)
    detected_all = sum(stats[k]["detected"] for k in hidden_keywords)
    overall_recall = (detected_all / total_all * 100.0) if total_all > 0 else 0.0
    
    detailed_recall = {k: (stats[k]["detected"] / stats[k]["total"] * 100.0) if stats[k]["total"] > 0 else 0.0 for k in hidden_keywords}
    
    return {
        "overall_recall": overall_recall,
        "total_hidden": total_all,
        "detected_hidden": detected_all,
        "detailed_recall": detailed_recall
    }


