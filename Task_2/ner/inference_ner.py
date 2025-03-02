from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification
import torch

class AnimalNER:
    def __init__(self, model_path="./ner_model/final_model"):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        self.model = DistilBertForTokenClassification.from_pretrained(model_path)
        self.animal_classes = ['cat', 'dog', 'horse', 'spider', 'butterfly',
                              'chicken', 'cow', 'sheep', 'elephant', 'squirrel']

    #Extracting animal data
    def extract_animal(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        outputs = self.model(**inputs).logits
        predictions = torch.argmax(outputs, dim=2)[0]
        
        #Converting to tokens
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        for token, pred in zip(tokens, predictions):
            if pred.item() == 1 and token.lower() in self.animal_classes:
                return token.lower()
        return None
    

