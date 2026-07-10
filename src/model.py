from transformers import RobertaForTokenClassification
from peft import get_peft_model, LoraConfig, TaskType

def get_roberta_lora_model(model_name="roberta-base", num_labels=19, r=8, lora_alpha=16, lora_dropout=0.1):
    """
    Load pre-trained RoBERTa model for token classification, freeze base weights,
    and inject LoRA parameters.
    """
    model = RobertaForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
    
    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        inference_mode=False,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["query", "value"]  # RoBERTa attention layers to adapt
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model
