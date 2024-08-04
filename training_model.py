import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import os

# Carregar os dados
dados = pd.read_csv(os.path.join("files", "sizing_messages.csv"))
print(dados['indicativo'].value_counts())
print(torch.cuda.is_available())

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(dados['mensagem'], dados['indicativo'], test_size=0.2, random_state=42)

# Inicializa o tokenizer BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def encode_data(texts, labels, tokenizer, max_length=64):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded_dict = tokenizer(
            text,
            add_special_tokens=True, 
            max_length=max_length,
            padding='max_length', 
            return_attention_mask=True, 
            return_tensors='pt', 
            truncation=True
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    labels = torch.tensor(labels.tolist())
    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0), labels

# Codifica os dados de treino e teste
train_inputs, train_masks, train_labels = encode_data(X_train, y_train, tokenizer)
test_inputs, test_masks, test_labels = encode_data(X_test, y_test, tokenizer)

# Define o batch size
batch_size = 16

# Cria os DataLoader para os dados de treino e teste
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

# Configura o dispositivo (GPU ou CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inicializa o modelo BERT para classificação
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(device)

# Configura o otimizador
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

# Número de épocas para treinamento
epochs = 10

# Listas para armazenar perdas e acurácia
training_loss = []
training_accuracy = []
validation_accuracy = []
f1_scores = []
precisions = []
recalls = []

# Configuração do Early Stopping
patience = 3
early_stopping_counter = 0
best_acc = 0

# Loop de treinamento
for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    model.train()

    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    for step, batch in enumerate(train_dataloader):
        batch_input_ids, batch_input_mask, batch_labels = tuple(t.to(device) for t in batch)

        model.zero_grad()

        outputs = model(batch_input_ids, token_type_ids=None, attention_mask=batch_input_mask, labels=batch_labels)
        loss = outputs.loss
        total_loss += loss.item()

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        correct_predictions += (predictions == batch_labels).sum().item()
        total_predictions += batch_labels.size(0)

        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_dataloader)
    train_acc = correct_predictions / total_predictions
    training_loss.append(avg_train_loss)
    training_accuracy.append(train_acc)

    # Avaliar no conjunto de teste
    model.eval()
    predictions, true_labels = [], []

    for batch in test_dataloader:
        batch_input_ids, batch_input_mask, batch_labels = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            outputs = model(batch_input_ids, token_type_ids=None, attention_mask=batch_input_mask)
        
        logits = outputs.logits
        predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
        true_labels.extend(batch_labels.cpu().numpy())

    acc = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    
    validation_accuracy.append(acc)
    f1_scores.append(f1)
    precisions.append(precision)
    recalls.append(recall)

    print(f"Training loss: {avg_train_loss}")
    print(f"Training accuracy: {train_acc}")
    print(f"Validation accuracy: {acc}")
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    # Early stopping
    if acc > best_acc:
        best_acc = acc
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1

    if early_stopping_counter >= patience:
        print("Early stopping")
        break

# Salvar o modelo e o tokenizer treinados
model_save_path = os.path.join('files', 'model')
tokenizer_save_path = os.path.join('files', 'tokenizer')

model.save_pretrained(model_save_path)
tokenizer.save_pretrained(tokenizer_save_path)

print("Training complete.")

# Salvar os dados de treinamento em um arquivo Excel
training_data = {
    'Epoch': list(range(1, len(training_loss) + 1)),
    'Training Loss': training_loss,
    'Training Accuracy': training_accuracy,
    'Validation Accuracy': validation_accuracy,
    'F1 Score': f1_scores,
    'Precision': precisions,
    'Recall': recalls
}

df_training_data = pd.DataFrame(training_data)
df_training_data.to_excel(os.path.join('files', 'training_metrics.xlsx'), index=False)

print("Training metrics saved to 'files/training_metrics.xlsx'.")
