import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForTokenClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
import numpy as np

# Animal classes
animal_classes = [
    "cat", "dog", "horse", "spider", "butterfly",
    "chicken", "cow", "sheep", "elephant", "squirrel"
]

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
    output_dir="./ner_model",
    evaluation_strategy="no",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
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
synthetic_data = generate_synthetic_data()
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
model.save_pretrained("./ner_model/final_model")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
tokenizer.save_pretrained("./ner_model/final_model")