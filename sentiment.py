import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
import math
import glob
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model, AdamW, get_linear_schedule_with_warmup
import wandb
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
import random

api_key = 'b574a868036a98ebf558d9abaf60f7fed3738a31'
# WandB setup
wandb.login(key=api_key)
wandb.init(project="model_sentiment1")

# Data loading functions
def load_writing_prompts() -> tuple:
    path = "writingPrompts"
    datasets = {}
    splits = ['train', 'valid', 'test']

    for split in splits:
        prompt_path = os.path.join(path, f"{split}.wp_source")
        story_path = os.path.join(path, f"{split}.wp_target")

        with open(prompt_path, 'r', encoding='utf-8') as prompt_file:
            prompts = [line.strip() for line in prompt_file]

        with open(story_path, 'r', encoding='utf-8') as story_file:
            stories = [line.strip() for line in story_file]

        datasets[split] = {'input_ids': prompts, 'attention_mask': stories, 'labels': stories}

        print(f"Loaded {split} data with {len(prompts)} prompts and {len(stories)} stories")

    return datasets['train'], datasets['valid'], datasets['test']

def load_imdb_dataset() -> tuple:
    imdb_path = "./aclImdb"
    datasets = {'train': [], 'test': []}

    for split in ['train', 'test']:
        for sentiment in ['pos', 'neg']:
            path = os.path.join(imdb_path, split, sentiment)
            files = glob.glob(os.path.join(path, "*.txt"))
            for file in files:
                with open(file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    label = 1 if sentiment == 'pos' else 0
                    datasets[split].append({'text': text, 'label': label})
    print("loaded sentiment dataset")

    return datasets['train'], datasets['test']

def custom_collate(batch):
    task = batch[0]['task']
    if task == 'story':
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels, 'task': task}
    
    elif task == 'sentiment':
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long)
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels, 'task': task}

class IMDBDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data['input_ids'])
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.data['input_ids'][idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.data['attention_mask'][idx], dtype=torch.long),
            'labels': torch.tensor(self.data['labels'][idx], dtype=torch.long),
            'task': 'sentiment'
        }

class WritingPromptsDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data['input_ids'])
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.data['input_ids'][idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.data['attention_mask'][idx], dtype=torch.long),
            'labels': torch.tensor(self.data['labels'][idx], dtype=torch.long),
            'task': 'story'
        }

class GPT2ForMultitask(nn.Module):
    def __init__(self, pretrained_model_name, num_sentiment_labels):
        super(GPT2ForMultitask, self).__init__()
        self.num_sentiment_labels = num_sentiment_labels
        self.gpt2 = GPT2Model.from_pretrained(pretrained_model_name)
        self.gpt2_lm_head = GPT2LMHeadModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(0.1)
        
        # Sentiment classification head
        self.classifier = nn.Linear(self.gpt2.config.hidden_size, num_sentiment_labels)
        
        # Story generation head
        self.lm_head = self.gpt2_lm_head.lm_head

    def forward(self, input_ids, attention_mask=None, labels=None, task='story'):
        outputs = self.gpt2(input_ids, attention_mask=attention_mask)
        hidden_state = outputs[0]
        
        if task == 'sentiment':
            pooled_output = hidden_state[:, -1, :]
            pooled_output = self.dropout(pooled_output)
            logits_sentiment = self.classifier(pooled_output)
            return logits_sentiment
        
        elif task == 'story':
            logits_next_token = self.lm_head(hidden_state)
            return logits_next_token


# Load data
train_data, valid_data, test_data = load_writing_prompts()
imdb_train_data, imdb_test_data = load_imdb_dataset()

# Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
tokenizer.pad_token = tokenizer.eos_token

# Tokenize IMDB data
inputs_train_imdb = tokenizer([item['text'] for item in imdb_train_data], padding='max_length', truncation=True, max_length=512)
inputs_test_imdb = tokenizer([item['text'] for item in imdb_test_data], padding='max_length', truncation=True, max_length=512)

attention_mask_train_imdb = torch.tensor(inputs_train_imdb['attention_mask'])
attention_mask_test_imdb = torch.tensor(inputs_test_imdb['attention_mask'])

labels_train_imdb = torch.tensor([item['label'] for item in imdb_train_data])
labels_test_imdb = torch.tensor([item['label'] for item in imdb_test_data])

train_dataset_imdb = IMDBDataset({'input_ids': inputs_train_imdb['input_ids'],
                                  'attention_mask': attention_mask_train_imdb,
                                  'labels': labels_train_imdb})

test_dataset_imdb = IMDBDataset({'input_ids': inputs_test_imdb['input_ids'],
                                 'attention_mask': attention_mask_test_imdb,
                                 'labels': labels_test_imdb})

# Tokenize story data
inputs_train = tokenizer(train_data['input_ids'], padding='max_length', truncation=True, max_length=512)
inputs_valid = tokenizer(valid_data['input_ids'], padding='max_length', truncation=True, max_length=512)

attention_mask_train = torch.tensor(inputs_train['attention_mask'])
attention_mask_valid = torch.tensor(inputs_valid['attention_mask'])

labels_train = torch.tensor(inputs_train['input_ids']).clone()
labels_train[inputs_train['input_ids'] == tokenizer.pad_token_id] = -100

labels_valid = torch.tensor(inputs_valid['input_ids']).clone()
labels_valid[inputs_valid['input_ids'] == tokenizer.pad_token_id] = -100

train_dataset = WritingPromptsDataset({'input_ids': inputs_train['input_ids'],
                                       'attention_mask': attention_mask_train,
                                       'labels': labels_train})

valid_dataset = WritingPromptsDataset({'input_ids': inputs_valid['input_ids'],
                                       'attention_mask': attention_mask_valid,
                                       'labels': labels_valid})

# DataLoaders
train_batch_size = 4
valid_batch_size = 4

train_dataloader_imdb = DataLoader(
    train_dataset_imdb,
    shuffle=True,
    batch_size=train_batch_size,
    collate_fn=custom_collate
)

train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=train_batch_size,
    collate_fn=custom_collate
)

valid_dataloader = DataLoader(
    valid_dataset,
    shuffle=False,
    batch_size=valid_batch_size,
    collate_fn=custom_collate
)

# Instantiate multitask model
num_sentiment_labels = 2  # For binary sentiment classification
model = GPT2ForMultitask('distilgpt2', num_sentiment_labels)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  
    r=4,  
    lora_alpha=32,  
    lora_dropout=0.1, 
    target_modules=["c_attn", "c_fc"]  
)

model = get_peft_model(model, lora_config)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Configuration
config = {
    "epochs": 1,
    "learning_rate_story": 0.01,
    "learning_rate_sentiment": 0.05,
    "batch_size": 4,
    "context_length": 512,
    "weight_decay_story": 0.01,
    "weight_decay_sentiment": 0.005,
    "adam_epsilon": 1e-8,
    "warmup_steps_story": 0.1,
    "warmup_steps_sentiment": 0.05,
    "alpha": 0.0  # Weight for sentiment loss scaling
}

wandb.config.update(config)

# Optimizers and schedulers
optimizer_story = AdamW(model.parameters(), lr=wandb.config.learning_rate_story, eps=wandb.config.adam_epsilon, weight_decay=wandb.config.weight_decay_story)
scheduler_story = get_linear_schedule_with_warmup(optimizer_story, num_warmup_steps=int(len(train_dataloader) * wandb.config.warmup_steps_story), num_training_steps=int(len(train_dataloader) * wandb.config.epochs))

optimizer_sentiment = AdamW(model.parameters(), lr=wandb.config.learning_rate_sentiment, eps=wandb.config.adam_epsilon, weight_decay=wandb.config.weight_decay_sentiment)
scheduler_sentiment = get_linear_schedule_with_warmup(optimizer_sentiment, num_warmup_steps=int(len(train_dataloader_imdb) * wandb.config.warmup_steps_sentiment), num_training_steps=int(len(train_dataloader_imdb) * wandb.config.epochs))

# Loss functions
loss_fct_story = nn.CrossEntropyLoss(ignore_index=-100)
loss_fct_sentiment = nn.CrossEntropyLoss()

import torch
from torch.cuda.amp import autocast, GradScaler

# Initialize scaler for mixed precision
scaler = GradScaler()

for epoch in range(wandb.config.epochs):
    model.train()
    total_train_loss = 0.0

    story_iterator = iter(train_dataloader)
    sentiment_iterator = iter(train_dataloader_imdb)

    for step in range(max(len(train_dataloader), len(train_dataloader_imdb))):
        # Story batch
        try:
            story_batch = next(story_iterator)
        except StopIteration:
            story_iterator = iter(train_dataloader)
            story_batch = next(story_iterator)
        
        story_input_ids = story_batch['input_ids'].to(device)
        story_attention_mask = story_batch['attention_mask'].to(device)
        story_labels = story_batch['labels'].to(device)

        model.zero_grad()
        with autocast():
            outputs_story, _ = model(input_ids=story_input_ids, attention_mask=story_attention_mask, labels=story_labels, task='story')
            loss_story = loss_fct_story(outputs_story.view(-1, model.gpt2.config.vocab_size), story_labels.view(-1))
        
        # Sentiment batch
        try:
            sentiment_batch = next(sentiment_iterator)
        except StopIteration:
            sentiment_iterator = iter(train_dataloader_imdb)
            sentiment_batch = next(sentiment_iterator)
        
        sentiment_input_ids = sentiment_batch['input_ids'].to(device)
        sentiment_attention_mask = sentiment_batch['attention_mask'].to(device)
        sentiment_labels = sentiment_batch['labels'].to(device)

        with autocast():
            outputs_sentiment, _ = model(input_ids=sentiment_input_ids, attention_mask=sentiment_attention_mask, labels=sentiment_labels, task='sentiment')
            loss_sentiment = loss_fct_sentiment(outputs_sentiment.view(-1, model.num_sentiment_labels), sentiment_labels.view(-1))
        
        # Combine losses
        total_loss = loss_story + wandb.config.alpha * loss_sentiment
        total_train_loss += total_loss.item()

        # Backpropagation with mixed precision
        scaler.scale(total_loss).backward()
        scaler.step(optimizer_story)
        scaler.step(optimizer_sentiment)
        scaler.update()

        # Scheduler step
        scheduler_story.step()
        scheduler_sentiment.step()

        # Log train loss
        wandb.log({"train_loss": total_loss.item()})

    # Logging average train loss per epoch
    average_train_loss = total_train_loss / len(train_dataloader)
    wandb.log({"epoch": epoch, "average_train_loss": average_train_loss})

    print(f'Epoch {epoch + 1}/{wandb.config.epochs} | Average Train Loss: {average_train_loss:.4f}')
    
    # Validation phase
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for val_batch in valid_dataloader:
            val_input_ids = val_batch['input_ids'].to(device)
            val_attention_mask = val_batch['attention_mask'].to(device)
            val_labels = val_batch['labels'].to(device)

            with autocast():
                outputs_val, _ = model(input_ids=val_input_ids, attention_mask=val_attention_mask, labels=val_labels, task='story')
                loss_val = loss_fct_story(outputs_val.view(-1, model.gpt2.config.vocab_size), val_labels.view(-1))
                total_val_loss += loss_val.item()

    # Calculate average validation loss
    average_val_loss = total_val_loss / len(valid_dataloader)
    wandb.log({"epoch": epoch, "average_val_loss": average_val_loss})

    print(f'Epoch {epoch + 1}/{wandb.config.epochs} | Average Validation Loss: {average_val_loss:.4f}')



def generate_story(prompt, max_length=512, num_return_sequences=1):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )
    stories = [tokenizer.decode(output, skip_special_tokens=True) for output in output_sequences]
    return stories

# Example usage:
prompt = "Once upon a time"
stories = generate_story(prompt)
for i, story in enumerate(stories):
    print(f"Story {i + 1}:\n{story}\n")