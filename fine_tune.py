import json
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from datasets import Dataset

# Load data
with open('scholarships.json', 'r') as f:
    data = json.load(f)

# Prepare training examples
train_examples = []
required_keys = ['NIT_admissions', 'IIT_admissions', 'political_power', 'education_consultancy', 'college_contact']

for key in required_keys:
    if key in data:
        for item in data[key]:
            train_examples.append(InputExample(texts=[item['question'], item['answer']]))
    else:
        print(f"Warning: Key '{key}' not found in the JSON file")

# Adding the scheme description to the training examples
if 'description' in data:
    train_examples.append(InputExample(texts=["What is COMPEX Scholarship?", data["description"]]))

# Load the model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Define a DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.MultipleNegativesRankingLoss(model)

# Fine-tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)

# Save the model
model.save('fine-tuned-model')
