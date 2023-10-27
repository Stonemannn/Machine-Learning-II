from transformers import pipeline
from transformers import AutoTokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline("sentiment-analysis", model=model_name)
result = classifier(["We are happy to show you the Transformers Library.","We hope you don't hate it."])
for res in result:
  print(res)