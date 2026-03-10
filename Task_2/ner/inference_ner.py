from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification
import torch
import argparse

class AnimalNER:
    def __init__(self, model_path="./ner_model/final_model"):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        self.model = DistilBertForTokenClassification.from_pretrained(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.model.eval()
        
        self.animal_classes = ['cat', 'dog', 'horse', 'spider', 'butterfly',
                              'chicken', 'cow', 'sheep', 'elephant', 'squirrel']

    #Extracting animal data
    def extract_animal(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2)[0]
        
        #Converting to tokens
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        for token, pred in zip(tokens, predictions):
            if pred.item() == 1 and token.lower() in self.animal_classes:
                return token.lower()
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract animal from text')
    parser.add_argument('--text', type=str, required=True,
                        help='Text to extract animal from')
    parser.add_argument('--model', type=str, default="./ner_model/final_model",
                        help='Path to trained NER model')
    args = parser.parse_args()
    
    ner = AnimalNER(model_path=args.model)
    animal = ner.extract_animal(args.text)
    print(f"Extracted animal: {animal if animal else 'None'}")
    

