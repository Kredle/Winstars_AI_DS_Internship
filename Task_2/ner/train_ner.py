import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForTokenClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train NER model for animal recognition')
    parser.add_argument('--num-examples', type=int, default=1000,
                        help='Number of synthetic training examples')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--output-dir', type=str, default="./ner_model",
                        help='Output directory for model')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    return parser.parse_args()

args = parse_args()

# Config from arguments
NUM_EXAMPLES = args.num_examples
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LR = args.lr
OUTPUT_DIR = args.output_dir

# Set seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Animal classes
animal_classes = [
    "cat", "dog", "horse", "spider", "butterfly",
    "chicken", "cow", "sheep", "elephant", "squirrel"
]

print(f"\nNER Training Configuration:")
print(f"  Synthetic Examples: {NUM_EXAMPLES}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Learning Rate: {LR}")
print(f"  Output Directory: {OUTPUT_DIR}")
print()

# Data generation
def generate_synthetic_data(num_examples=1000):
    texts = []
    labels = []
    templates = [
        "There is a {} in the picture",
        "I can see a {} here",
        "This image contains a {}",
        "Look at this {}!",
        "Is that a {}?"
    ]
    
    for _ in range(num_examples):
        animal = np.random.choice(animal_classes)
        template = np.random.choice(templates)
        text = template.format(animal)
        
        # Fiding animal position
        start_idx = text.find(animal)
        end_idx = start_idx + len(animal)
        
        # Creating labels
        bio_labels = ["O"] * len(text.split())
        for i, word in enumerate(text.split()):
            if word == animal:
                bio_labels[i] = "B-ANIMAL"
        texts.append(text)
        labels.append(bio_labels)
    
    return {"text": texts, "labels": labels}

#Tokenizing 
def tokenize_and_align_labels(examples):
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    tokenized_inputs = tokenizer(
        examples["text"],
        truncation=True,
        is_split_into_words=False,
        padding="max_length",
        max_length=128
    )
    
    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                # -100 
                label_ids.append(-100)
            elif word_idx >= len(label):
                # Ignore tokens if they are out of range
                label_ids.append(-100)
            else:
                if word_idx != previous_word_idx:
                    # First word_idx is getting a label
                    label_ids.append(1 if label[word_idx] == "B-ANIMAL" else 0)
                else:
                    # Ignore rest
                    label_ids.append(-100)
                previous_word_idx = word_idx
        
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Training settings
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LR,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
)

# Model initialization
model = DistilBertForTokenClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,
    id2label={0: "O", 1: "B-ANIMAL"},
    label2id={"O": 0, "B-ANIMAL": 1}
)

# Getting dataset
synthetic_data = generate_synthetic_data(num_examples=NUM_EXAMPLES)
dataset = Dataset.from_dict(synthetic_data)
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# Training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()

# Saving the model
final_model_path = f"{OUTPUT_DIR}/final_model"
model.save_pretrained(final_model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
tokenizer.save_pretrained(final_model_path)
print(f"\nTraining complete! Model saved to {final_model_path}")