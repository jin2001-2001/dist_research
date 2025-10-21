from transformers import BertForMaskedLM, BertTokenizer

tok = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")
print(model.cls.predictions)