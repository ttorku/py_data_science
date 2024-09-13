# fine_tune_flan_t5_small_lora.py

# Import Libraries
import pandas as pd
import torch
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from rouge_score import rouge_scorer
from peft import LoraConfig, get_peft_model

# Load and Preprocess Data
# Replace 'your_dataset.csv' with your actual dataset file
df = pd.read_csv('your_dataset.csv')

# If your components are in separate columns, combine them
# Uncomment and modify the following line if needed
# df['output'] = df[['comp1', 'comp2', 'comp3', 'comp4', 'comp5', 'comp6']].apply(lambda x: ' <SEP> '.join(x), axis=1)

# Split into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

# Convert to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))

# Tokenizer and Tokenization
tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-small')
tokenizer.add_special_tokens({'sep_token': '<SEP>'})

def preprocess_function(examples):
    inputs = examples['input']
    targets = examples['output']
    model_inputs = tokenizer(
        inputs,
        max_length=256,
        padding='max_length',
        truncation=True,
    )
    labels = tokenizer(
        targets,
        max_length=256,
        padding='max_length',
        truncation=True,
    )
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_val = val_dataset.map(preprocess_function, batched=True)

# Set format for PyTorch
tokenized_train.set_format(
    type='torch',
    columns=['input_ids', 'attention_mask', 'labels'],
)
tokenized_val.set_format(
    type='torch',
    columns=['input_ids', 'attention_mask', 'labels'],
)

# Load Model and Apply LoRA
model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-small')
model.resize_token_embeddings(len(tokenizer))

# Define LoRA Config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=['q', 'v'],
    lora_dropout=0.1,
    bias='none',
    task_type='SEQ_2_SEQ_LM',
)

# Apply LoRA
model = get_peft_model(model, lora_config)

# Training Arguments
training_args = TrainingArguments(
    output_dir='./flan-t5-small-finetuned-lora',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_steps=500,
    save_steps=500,
    evaluation_strategy="steps",
    logging_dir='./logs_flan_t5_lora',
    logging_steps=100,
    save_total_limit=2,
    # fp16=True,  # Uncomment this line to enable mixed precision training (requires compatible GPU)
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
trainer.save_model('./flan-t5-small-finetuned-lora')

# Inference and Evaluation with ROUGE Scores
# Prepare Test Data
test_df = val_df.copy()

# Initialize ROUGE Scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

def generate_output(model, tokenizer, input_text):
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    # Uncomment the next line if using GPU
    # input_ids = input_ids.to('cuda')
    output_ids = model.generate(
        input_ids=input_ids,
        max_length=256,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True,
    )
    # Uncomment the next line if using GPU
    # output_ids = output_ids.to('cpu')
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

# Load the Fine-tuned Model
model = T5ForConditionalGeneration.from_pretrained('./flan-t5-small-finetuned-lora')
model.resize_token_embeddings(len(tokenizer))
model = get_peft_model(model, lora_config)
# Uncomment the next line if using GPU
# model.to('cuda')
model.eval()

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
