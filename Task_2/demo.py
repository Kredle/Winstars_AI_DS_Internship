#!/usr/bin/env python3
"""
Comprehensive demo script for the Animal Verification Pipeline.
Shows how to:
1. Train the image classifier
2. Train the NER model
3. Run inference on images and text
4. Use the full verification pipeline
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Adding paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ner.inference_ner import AnimalNER
from image_classifier.inference_classifier import AnimalClassifier
from pipeline.main_pipeline import VerificationPipeline


def demo_train_classifier(args):
    """Demo: Train image classifier with custom parameters"""
    print("\n" + "="*70)
    print("DEMO 1: Training Image Classifier")
    print("="*70)
    
    cmd = [
        "python", "image_classifier/train_classifier.py",
        "--data-path", args.data_path,
        "--batch-size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--output-model", args.classifier_model,
        "--best-model", args.best_model,
    ]
    
    print(f"\nRunning: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=args.root_dir)
    
    if result.returncode == 0:
        print(f"Classifier training complete! Model: {args.classifier_model}")
    else:
        print(f"Training failed with code {result.returncode}")
    
    return result.returncode == 0


def demo_train_ner(args):
    """Demo: Train NER model with custom parameters"""
    print("\n" + "="*70)
    print("DEMO 2: Training NER Model")
    print("="*70)
    
    cmd = [
        "python", "ner/train_ner.py",
        "--num-examples", str(args.ner_examples),
        "--epochs", str(args.ner_epochs),
        "--batch-size", str(args.ner_batch_size),
        "--lr", str(args.ner_lr),
        "--output-dir", args.ner_model_dir,
    ]
    
    print(f"\nRunning: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=args.root_dir)
    
    if result.returncode == 0:
        print(f"NER training complete! Model: {args.ner_model_dir}")
    else:
        print(f"Training failed with code {result.returncode}")
    
    return result.returncode == 0


def demo_inference_classifier(args):
    """Demo: Run image classifier inference"""
    print("\n" + "="*70)
    print("DEMO 3: Image Classification Inference")
    print("="*70)
    
    # Finding a sample image
    sample_image = None
    data_path = Path(args.data_path)
    if data_path.exists():
        for class_dir in data_path.iterdir():
            if class_dir.is_dir():
                images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg"))
                if images:
                    sample_image = str(images[0])
                    break
    
    if not sample_image:
        print("No sample images found. Skipping inference demo.")
        return False
    
    print(f"\nUsing sample image: {sample_image}")
    print(f"Loading model from: {args.classifier_model}\n")
    
    try:
        classifier = AnimalClassifier(model_path=args.classifier_model)
        prediction = classifier.predict(sample_image)
        print(f"Prediction: {prediction}")
        return True
    except Exception as e:
        print(f"Inference failed: {e}")
        return False


def demo_inference_ner(args):
    """Demo: Run NER inference"""
    print("\n" + "="*70)
    print("DEMO 4: Named Entity Recognition (NER)")
    print("="*70)
    
    test_texts = [
        "I can see a dog in the image",
        "There is a beautiful cat here",
        "Look at the elephant",
    ]
    
    print(f"\nLoading NER model from: {args.ner_model_dir}\n")
    
    try:
        ner = AnimalNER(model_path=f"{args.ner_model_dir}/final_model")
        
        for text in test_texts:
            animal = ner.extract_animal(text)
            print(f"Text: \"{text}\"")
            print(f"  => Extracted: {animal}\n")
        
        print("NER inference complete")
        return True
    except Exception as e:
        print(f"NER inference failed: {e}")
        return False


def demo_pipeline(args):
    """Demo: Full verification pipeline"""
    print("\n" + "="*70)
    print("DEMO 5: Full Verification Pipeline")
    print("="*70)
    
    # Find a sample image
    sample_image = None
    data_path = Path(args.data_path)
    if data_path.exists():
        for class_dir in data_path.iterdir():
            if class_dir.is_dir():
                images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg"))
                if images:
                    sample_image = str(images[0])
                    break
    
    if not sample_image:
        print("No sample images found. Skipping pipeline demo.")
        return False
    
    test_cases = [
        ("There is a dog in the image", sample_image),
        ("I spotted a beautiful cat", sample_image),
    ]
    
    print(f"\nLoading models...")
    print(f"  - Classifier: {args.classifier_model}")
    print(f"  - NER: {args.ner_model_dir}\n")
    
    try:
        pipeline = VerificationPipeline(
            classifier_path=args.classifier_model,
            ner_path=f"{args.ner_model_dir}/final_model"
        )
        
        for text, image_path in test_cases:
            print(f"Text: \"{text}\"")
            print(f"Image: {image_path}")
            
            try:
                result = pipeline.verify(text, image_path)
                print(f"  => Match: {'Yes' if result else 'No'}\n")
            except Exception as e:
                print(f"  => Error: {e}\n")
        
        print("Pipeline demo complete")
        return True
    except Exception as e:
        print(f"Pipeline demo failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Animal Verification Pipeline - Comprehensive Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all demos
  python demo.py --all
  
  # Run specific training demos
  python demo.py --train-classifier --train-ner
  
  # Run only inference demos
  python demo.py --infer-classifier --infer-ner --pipeline
  
  # Custom parameters
  python demo.py --train-classifier --epochs 10 --batch-size 64
        """
    )
    
    # Demo selection
    parser.add_argument('--all', action='store_true',
                        help='Run all demos')
    parser.add_argument('--train-classifier', action='store_true',
                        help='Train image classifier')
    parser.add_argument('--train-ner', action='store_true',
                        help='Train NER model')
    parser.add_argument('--infer-classifier', action='store_true',
                        help='Run image classifier inference')
    parser.add_argument('--infer-ner', action='store_true',
                        help='Run NER inference')
    parser.add_argument('--pipeline', action='store_true',
                        help='Run full verification pipeline')
    
    # Classifier parameters
    parser.add_argument('--data-path', type=str, default="./data/animals-10/raw-img",
                        help='Path to dataset')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for classifier training')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Epochs for classifier training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for classifier')
    parser.add_argument('--classifier-model', type=str, default="animal_classifier.pth",
                        help='Path to save/load classifier model')
    parser.add_argument('--best-model', type=str, default="best_model.pth",
                        help='Path to save best classifier model')
    
    # NER parameters
    parser.add_argument('--ner-examples', type=int, default=1000,
                        help='Number of synthetic examples for NER training')
    parser.add_argument('--ner-epochs', type=int, default=3,
                        help='Epochs for NER training')
    parser.add_argument('--ner-batch-size', type=int, default=16,
                        help='Batch size for NER training')
    parser.add_argument('--ner-lr', type=float, default=2e-5,
                        help='Learning rate for NER')
    parser.add_argument('--ner-model-dir', type=str, default="./ner_model",
                        help='Path to NER model directory')
    
    # Miscellaneous
    parser.add_argument('--root-dir', type=str, default=".",
                        help='Root directory for running commands')
    
    args = parser.parse_args()
    
    # If no demo specified, showing tip to the user
    if not any([args.all, args.train_classifier, args.train_ner, 
                args.infer_classifier, args.infer_ner, args.pipeline]):
        parser.print_help()
        print("\nTip: Run 'python demo.py --all' to execute all demos")
        return
    
    print("\n" + "="*35)
    print("           ANIMAL VERIFICATION PIPELINE - DEMO")
    print("="*35 + "\n")
    
    results = {}
    
    # Training demos
    if args.all or args.train_classifier:
        results['train_classifier'] = demo_train_classifier(args)
    
    if args.all or args.train_ner:
        results['train_ner'] = demo_train_ner(args)
    
    # Inference demos
    if args.all or args.infer_classifier:
        results['infer_classifier'] = demo_inference_classifier(args)
    
    if args.all or args.infer_ner:
        results['infer_ner'] = demo_inference_ner(args)
    
    if args.all or args.pipeline:
        results['pipeline'] = demo_pipeline(args)
    
    # Summary
    print("\n" + "="*70)
    print("DEMO SUMMARY")
    print("="*70)
    for demo_name, success in results.items():
        status = "PASS" if success else "FAIL"
        demo_label = demo_name.replace('_', ' ').title()
        print(f"{demo_label:<30} {status}")
    
    print("\n" + "="*35 + "\n")


if __name__ == "__main__":
    main()
