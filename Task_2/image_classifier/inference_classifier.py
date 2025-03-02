import torch
from torchvision import transforms, models
from PIL import Image

class AnimalClassifier:
    def __init__(self, model_path="animal_classifier.pth"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Model creating
        self.model = models.resnet18(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 10)  # 10 классов
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
            'cat', 'dog', 'horse', 'spider', 'butterfly',
            'chicken', 'cow', 'sheep', 'elephant', 'squirrel'
        ]

    #Predict function
    def predict(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image)
            _, pred = torch.max(outputs, 1)
        
        return self.classes[pred.item()]

