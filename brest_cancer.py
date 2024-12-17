from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

import torch
import torch.nn as nn

np.random.seed(123)
torch.manual_seed(123)

previsores = pd.read_csv("data\entradas_breast.csv")
classes = pd.read_csv("data\saidas_breast.csv")

previsores_treinamento, previsores_teste, classes_treinamento, classes_teste = train_test_split(previsores, classes, test_size=0.25)

previsores_treinamento = torch.tensor(np.array(previsores_treinamento, dtype=np.float32))
classes_treinamento = torch.tensor(np.array(classes_treinamento, dtype=np.float32))

dataset = torch.utils.data.TensorDataset(previsores_treinamento, classes_treinamento)

train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=10, shuffle=True)

classificador = nn.Sequential(
  nn.Linear(in_features=30, out_features=16),
  nn.ReLU(),
  nn.Linear(16, 16),
  nn.ReLU(),
  nn.Linear(16, 1),
  nn.Sigmoid(),
)

criterion = nn.BCELoss()

optmizer = torch.optim.Adam(classificador.parameters(), lr=0.001, weight_decay=0.0001)

for epoch in range(100):
  running_loss = 0.

  for data in train_loader:
    inputs, labels = data

    optmizer.zero_grad()

    outputs = classificador(inputs)

    loss = criterion(outputs, labels)

    loss.backward()
    optmizer.step()

    running_loss += loss.item()
  
  print('Época %3d: perda %.5f' % (epoch + 1, running_loss / len(train_loader)))



classificador.eval()

previsores_teste = torch.tensor(np.array(previsores_teste, dtype=np.float32), dtype=torch.float32)

previsoes = classificador(previsores_teste)
previsoes = np.array(previsoes > 0.5)

taxa_acerto = accuracy_score(classes_teste, previsoes)
print(taxa_acerto)