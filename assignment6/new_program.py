from transformers import pipeline

unmasker = pipeline('fill-mask', model='bert-base-uncased')

results = unmasker("Hello I'm [MASK].")

for res in results:
    print(res)