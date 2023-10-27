from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm
import evaluate
from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler
import os
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

if __name__ == '__main__':
    # define Data, checkpoint, and tokenizer
    # GLUE (General Language Understanding Evaluation)
    # MRPC stands for Microsoft Research Paraphrase Corpus.
    raw_datasets = load_dataset("glue", "mrpc")
    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # hyper parameters
    batch_size = 4
    num_epochs = 3
    learning_rate = 5e-5
    num_labels = 2
    num_workers = 4
    training_length = 50

    # tokenizer
    def tokenize_function(example):
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    # reform the data for fine-tuning
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    # define data loader DataCollatorWithPadding helps to collate multiple data samples into a single batch for
    # training or evaluation. It takes care of dynamically padding your inputs to the same length so that they can be
    # processed together in a single batch.
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=batch_size, collate_fn=data_collator, num_workers=num_workers)
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], batch_size=batch_size, collate_fn=data_collator, num_workers=num_workers)


    for batch in train_dataloader:
        break
    {k: v.shape for k, v in batch.items()}

    # define model
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)

    # define optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # outputs = model(**batch)
    # print(outputs.loss, outputs.logits.shape)

    # num_training_steps = num_epochs * len(train_dataloader)
    num_training_steps = num_epochs * training_length
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    print(num_training_steps)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # tqdm displays a progress bar in the console (or notebook) to provide a visual indicator of how far a loop has progressed.
    # without tqdm, I won't be able to track the progress of the loop.
    progress_bar = tqdm(range(num_training_steps))

    #training
    model.train()
    for epoch in range(num_epochs):

        break_training_count = 0

        for batch in train_dataloader:

            break_training_count = break_training_count + 1
            if break_training_count > training_length:
                continue

            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    print('')
    print('The training process of the finetuning is finished')
    print('Now is working on evaluation ... ')

    # evaluation
    metric = evaluate.load("glue", "mrpc")
    model.eval()
    progress_bar = tqdm(range(len(eval_dataloader)))
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
        progress_bar.update(1)

    eval_results = metric.compute()
    print('')
    print('The evaluation process of the finetuning is finished, the results are')
    print(eval_results)

