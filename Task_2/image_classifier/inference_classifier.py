import torch
from torchvision import transforms, models
from PIL import Image
import argparse

class AnimalClassifier:
    def __init__(self, model_path="animal_classifier.pth", num_classes=10):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Model creating
        self.model = models.resnet18(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval() 
        
        # Image composing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # List of animal classes
        self.classes = [
            'butterfly', 'cat', 'chicken', 'cow', 'dog', 
            'elephant', 'horse', 'sheep', 'spider', 'squirrel'
        ]

    #Predict function
    def predict(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"Could not load image from {image_path}: {e}")
        
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image)
            _, pred = torch.max(outputs, 1)
        
        return self.classes[pred.item()]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify animal in image')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to image file')
    parser.add_argument('--model', type=str, default="animal_classifier.pth",
                        help='Path to trained model')
    args = parser.parse_args()
    
    classifier = AnimalClassifier(model_path=args.model)
    prediction = classifier.predict(args.image)
    print(f"Predicted animal: {prediction}")

