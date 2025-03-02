import sys
import os
import argparse

# Adding the path to root dir
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ner.inference_ner import AnimalNER
from image_classifier.inference_classifier import AnimalClassifier

class VerificationPipeline:
    def __init__(self):
        self.ner = AnimalNER()
        self.classifier = AnimalClassifier()

    def verify(self, text, image_path):
        #Getting the right animal
        animal_from_text = self.ner.extract_animal(text)
        #Making a prediction
        animal_from_image = self.classifier.predict(image_path)
        
        #If no animal class in --text
        if not animal_from_text:
            return False
        return animal_from_text == animal_from_image.lower()

if __name__ == "__main__":

    #Parsing data
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()
    
    pipeline = VerificationPipeline()
    result = pipeline.verify(args.text, args.image)
    print(f"Result: {result}")