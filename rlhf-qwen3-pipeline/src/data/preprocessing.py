from transformers import AutoTokenizer

def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def preprocess_data(examples, tokenizer, max_length=512):
    tokenized_inputs = tokenizer(
        examples['text'],
        truncation=True,
        max_length=max_length,
        padding='max_length',
        return_tensors='pt'
    )
    return tokenized_inputs

def format_dataset(dataset, tokenizer):
    processed_data = []
    for example in dataset:
        tokenized_example = preprocess_data(example, tokenizer)
        processed_data.append(tokenized_example)
    return processed_data

def save_preprocessed_data(processed_data, file_path):
    import torch
    torch.save(processed_data, file_path)