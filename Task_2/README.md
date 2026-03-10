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
- **Parameterizable training/inference scripts** (NEW)
- **Comprehensive exploratory data analysis** (NEW)
- **Full-featured demo script** (NEW)

## Project Structure 
```markdown
Animal-Verification/
├── data/
│ └── animals-10/ # Dataset directory
├── notebooks/
│ └── data_analysis.ipynb # Comprehensive EDA (class distribution, dimensions, colors)
├── ner/
│ ├── train_ner.py # NER model training (parameterizable)
│ └── inference_ner.py # Text entity extraction
├── image_classifier/
│ ├── train_classifier.py # Vision model training (parameterizable)
│ └── inference_classifier.py # Image prediction
├── pipeline/
│ └── main_pipeline.py # Combined verification system
├── models/
│ ├── best_model.pth # Best performing weights
│ └── animal_classifier.pth # Final classifier
├── demo.py # Comprehensive demo script (NEW)
├── requirements.txt # Dependencies
└── README.md # This file
```

# Download dataset (from Kaggle)
```bash
kaggle datasets download -d alessiocorrado99/animals10
unzip animals10.zip -d data/animals-10 #Then rename its folders 
```

## Usage Examples

### 1. Exploratory Data Analysis
Open `notebooks/data_analysis.ipynb` in Jupyter or VS Code to explore:
- Class distribution (10 animal types)
- Sample images per class (4 random samples)
- Image dimensions & aspect ratios
- RGB channel statistics per class

### 2. Training Models (Parameterizable)

#### Train Image Classifier
```bash
# Basic (defaults: 20 epochs, batch size 32, lr 0.001)
python image_classifier/train_classifier.py

# With custom parameters
python image_classifier/train_classifier.py \
    --data-path "./data/animals-10/raw-img" \
    --epochs 15 \
    --batch-size 64 \
    --lr 0.0005 \
    --output-model "my_classifier.pth"
```

**Available parameters:**
```
--data-path         Path to dataset (default: ./data/animals-10/raw-img)
--batch-size        Batch size (default: 32)
--epochs            Number of epochs (default: 20)
--num-classes       Number of classes (default: 10)
--val-split         Validation split (default: 0.2)
--lr                Learning rate (default: 0.001)
--output-model      Model output path (default: animal_classifier.pth)
--best-model        Best model path (default: best_model.pth)
--seed              Random seed (default: 42)
```

#### Train NER Model
```bash
# Basic (defaults: 3 epochs, batch size 16, 1000 synthetic examples)
python ner/train_ner.py

# With custom parameters
python ner/train_ner.py \
    --num-examples 2000 \
    --epochs 5 \
    --batch-size 32 \
    --lr 3e-5 \
    --output-dir "./my_ner_model"
```

**Available parameters:**
```
--num-examples      Synthetic training examples (default: 1000)
--epochs            Number of epochs (default: 3)
--batch-size        Batch size (default: 16)
--lr                Learning rate (default: 2e-5)
--output-dir        Output directory (default: ./ner_model)
--seed              Random seed (default: 42)
```

### 3. Inference (Parameterizable)

#### Image Classification
```bash
python image_classifier/inference_classifier.py \
    --image "data/animals-10/cat/sample.jpg" \
    --model "animal_classifier.pth"
```

#### Text-to-Animal Extraction (NER)
```bash
python ner/inference_ner.py \
    --text "I can see a beautiful dog in the image" \
    --model "./ner_model/final_model"
```

### 4. Full Pipeline Verification
```bash
python pipeline/main_pipeline.py \
    --text "There is a black cat in the image" \
    --image "data/animals-10/cat/2.jpeg" 
```

### 5. Comprehensive Demo (NEW)
Run all demos or specific ones:

```bash
# Run all demos
python demo.py --all

# Run specific demos
python demo.py --train-classifier --infer-classifier

# Train with custom parameters (demo)
python demo.py --train-classifier --epochs 5 --batch-size 64

# Train NER only
python demo.py --train-ner --ner-examples 500

# Full pipeline demo
python demo.py --pipeline

# Run inference only
python demo.py --infer-classifier --infer-ner --pipeline
```

**Demo options:**
```
--all                  Run all demos
--train-classifier    Train image classifier
--train-ner           Train NER model
--infer-classifier    Test classifier inference
--infer-ner           Test NER inference
--pipeline            Run full pipeline demo
```

### Enhanced Exploratory Data Analysis
The `notebooks/data_analysis.ipynb` now includes comprehensive analysis:
- **Class Distribution**: Bar charts, pie charts, and imbalance ratio calculation
- **Sample Visualization**: 4 random samples from each animal class
- **Dimension Analysis**: Width, height, and aspect ratio distributions
- **Color Statistics**: RGB channel mean and std per class
- **Pixel Intensities**: Grayscale distribution overlay

### Parameterizable Training Scripts
Both training scripts now accept command-line arguments for easy experimentation:
- **Image Classifier**: Control batch size, epochs, learning rate, output paths
- **NER Model**: Control synthetic examples, epochs, batch size, learning rate
- All scripts support custom random seeds for reproducibility
- Configuration printed at training start for easy logging

### Comprehensive Demo Script
The `demo.py` script provides end-to-end demonstrations:
- **run `--all`**: Execute all demos automatically
- **Individual demos**: Test training, inference, and pipeline separately
- **Custom parameters**: Pass hyperparameters to demos for quick experimentation
- **Error handling**: Graceful failure reporting
- **Results summary**: Clear pass/fail status for each demo

### Improved Inference Interfaces
All inference scripts now support command-line execution:
```bash
# Direct CLI usage of inference modules
python image_classifier/inference_classifier.py --image path.jpg --model model.pth
python ner/inference_ner.py --text "your text" --model ./model
python pipeline/main_pipeline.py --text "text" --image path.jpg --ner-model ... --classifier-model ...
```
