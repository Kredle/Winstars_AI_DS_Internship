import sys
import os
import argparse

# Adding the path to root dir
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ner.inference_ner import AnimalNER
from image_classifier.inference_classifier import AnimalClassifier

class VerificationPipeline:
    def __init__(self, ner_path="./ner_model/final_model", classifier_path="animal_classifier.pth"):
        self.ner = AnimalNER(model_path=ner_path)
        self.classifier = AnimalClassifier(model_path=classifier_path)

    def verify(self, text, image_path):
        # Getting the right animal
        animal_from_text = self.ner.extract_animal(text)
        # Making a prediction
        animal_from_image = self.classifier.predict(image_path)
        print(f"NER extracted: {animal_from_text}")
        print(f"Classifier predicted: {animal_from_image}")
        # If no animal class in --text
        if not animal_from_text:
            return False
        return animal_from_text == animal_from_image.lower()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Verify animal in text matches image')
    parser.add_argument("--text", type=str, required=True,
                        help="Text description of the animal")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to image file")
    parser.add_argument("--ner-model", type=str, default="./ner_model/final_model",
                        help="Path to NER model")
    parser.add_argument("--classifier-model", type=str, default="animal_classifier.pth",
                        help="Path to classifier model")
    args = parser.parse_args()

    
    try:
        pipeline = VerificationPipeline(
            ner_path=args.ner_model,
            classifier_path=args.classifier_model
        )
        result = pipeline.verify(args.text, args.image)
        print(f"\nResult: Match" if result else "\nResult: No match")
    except Exception as e:
        print(f"\nError: {e}")