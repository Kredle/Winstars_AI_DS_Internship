# Animal Verification Pipeline 

A Pipeline that verifies if a text description matches image content using:
- **Named Entity Recognition (NER)** with DistilBERT
- **Image Classification** with ResNet18

## Features 
- Dual-modality verification (text + image)
- Transformer-based entity extraction
- Fine-tuned vision model (10 animal classes)
- Class imbalance handling
- Validation metrics tracking
- GPU acceleration support

## Project Structure 
```markdown
Animal-Verification/
├── data/
│ └── animals-10/ # Dataset directory
├── notebooks/
│ └── EDA.ipynb # Data analysis notebook
├── ner/
│ ├── train_ner.py # NER model training
│ └── inference_ner.py # Text entity extraction
├── image_classifier/
│ ├── train_classifier.py # Vision model training
│ └── inference_classifier.py # Image prediction
├── pipeline/
│ └── main_pipeline.py # Combined verification system
├── models/
│ ├── best_model.pth # Best performing weights
│ └── animal_classifier.pth # Final classifier
└── requirements.txt # Dependencies
```

# Download dataset (from Kaggle)
```bash
kaggle datasets download -d alessiocorrado99/animals10
unzip animals10.zip -d data/animals-10 #Then rename its folders 
```

## Usage 
Full Pipeline Verification
```bash
python pipeline/main_pipeline.py \
    --text "There is a black cat in the image" \
    --image "data/animals-10/cat/2.jpeg"
```
## Output example:
```bash
Result: True
```
    
## Component Usage
1. Text Processing (NER):
```python
from ner.inference_ner import AnimalNER
ner = AnimalNER()
animal = ner.extract_animal("I spotted a grazing cow")  # Returns "cow"
```

3. Image Classification:
```python
from image_classifier.inference_classifier import AnimalClassifier
classifier = AnimalClassifier()
prediction = classifier.predict("data/animals-10/cat/1.jpeg")  # Returns "cat"
```

## Training Models 
1. Train NER Model
```bash
python ner/train_ner.py
```
- Generates synthetic training data
- Trains for 3 epochs with batch size 16
- Saves model to ner_model/final_model
2. Train Image Classifier
```bash
python image_classifier/train_classifier.py
```
- 20 training epochs
- 20% validation split
- Automatic class weighting
- Learning rate scheduling
- Saves best model to best_model.pth
