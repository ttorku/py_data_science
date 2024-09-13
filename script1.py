# fine_tune_gpt2_medium.py

# Import Libraries
import pandas as pd
import torch
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from rouge_score import rouge_scorer

# Load and Preprocess Data
# Replace 'your_dataset.csv' with your actual dataset file
df = pd.read_csv('your_dataset.csv')

# If your components are in separate columns, combine them
# Uncomment and modify the following line if needed
# df['output'] = df[['comp1', 'comp2', 'comp3', 'comp4', 'comp5', 'comp6']].apply(lambda x: ' <SEP> '.join(x), axis=1)

# Split into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.1)

# Convert to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))

# Tokenizer and Tokenization
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
tokenizer.add_special_tokens({'pad_token': '[PAD]', 'sep_token': '<SEP>'})

def tokenize_function(example):
    inputs = tokenizer(
        example['input'],
        padding='max_length',
        truncation=True,
        max_length=256,
    )
    outputs = tokenizer(
        example['output'],
        padding='max_length',
        truncation=True,
        max_length=256,
    )
    inputs['labels'] = outputs['input_ids']
    return inputs

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
tokenized_train.set_format(
    type='torch',
    columns=['input_ids', 'attention_mask', 'labels'],
)
tokenized_val.set_format(
    type='torch',
    columns=['input_ids', 'attention_mask', 'labels'],
)

# Load Model
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
model.resize_token_embeddings(len(tokenizer))

# Training Arguments
training_args = TrainingArguments(
    output_dir='./gpt2-medium-finetuned',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    eval_steps=500,
    save_steps=500,
    evaluation_strategy="steps",
    logging_dir='./logs',
    logging_steps=100,
    save_total_limit=2,
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
)

# Train the Model
trainer.train()

# Save the Model
trainer.save_model('./gpt2-medium-finetuned')

# Inference and Evaluation with ROUGE Scores
# Prepare Test Data
test_df = val_df.copy()

# Initialize ROUGE Scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

def generate_output(model, tokenizer, input_text):
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output_ids = model.generate(
        input_ids=input_ids,
        max_length=256,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def compute_rouge_scores(reference, prediction):
    # Split reference and prediction into components
    ref_components = [comp.strip() for comp in reference.split('<SEP>')]
    pred_components = [comp.strip() for comp in prediction.split('<SEP>')]

    # Ensure both have the same number of components
    min_len = min(len(ref_components), len(pred_components))
    ref_components = ref_components[:min_len]
    pred_components = pred_components[:min_len]

    scores = []
    for ref, pred in zip(ref_components, pred_components):
        score = scorer.score(ref, pred)
        scores.append({
            'reference': ref,
            'prediction': pred,
            'rouge1_f1': score['rouge1'].fmeasure,
            'rougeL_f1': score['rougeL'].fmeasure
        })
    return scores

# Run Inference and Evaluation
results = []

for idx, row in test_df.iterrows():
    input_text = row['input']
    reference_output = row['output']
    predicted_output = generate_output(model, tokenizer, input_text)
    component_scores = compute_rouge_scores(reference_output, predicted_output)
    results.append({
        'input': input_text,
        'reference': reference_output,
        'prediction': predicted_output,
        'component_scores': component_scores
    })

# Display Results
for i, result in enumerate(results):
    print(f"Sample {i+1}:")
    print(f"Input: {result['input']}\n")
    print("Reference Components:")
    ref_components = [comp.strip() for comp in result['reference'].split('<SEP>')]
    for j, comp in enumerate(ref_components):
        print(f"  Component {j+1}: {comp}")

    print("\nGenerated Components:")
    pred_components = [comp.strip() for comp in result['prediction'].split('<SEP>')]
    for j, comp in enumerate(pred_components):
        print(f"  Component {j+1}: {comp}")

    print("\nROUGE Scores per Component:")
    for j, comp_score in enumerate(result['component_scores']):
        print(f"Component {j+1}:")
        print(f"  Actual Component: {comp_score['reference']}")
        print(f"  Generated Component: {comp_score['prediction']}")
        print(f"  ROUGE-1 F1 Score: {comp_score['rouge1_f1']:.4f}")
        print(f"  ROUGE-L F1 Score: {comp_score['rougeL_f1']:.4f}")
        print()
    print("="*50)
