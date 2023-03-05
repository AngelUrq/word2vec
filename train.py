import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

embedding_size = 300
context_window = 5
batch_size = 1024

word_dataset = WordsDataset(context_window, context_first=True)
dataloader = DataLoader(word_dataset, batch_size=batch_size, shuffle=True)

print(f'Number of samples: {len(word_dataset.dataset["text"])}')
model = CBoW(len(word_dataset.vocab), embedding_size).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

for epoch in range(150):
  for batch, (X, y) in enumerate(dataloader):
    X = X.squeeze().to(device)
    y = F.one_hot(y, num_classes=len(word_dataset.vocab)).float().to(device)

    # Compute prediction error
    pred = model(X)

    loss = loss_fn(pred, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch % 100 == 0:
      accuracy = (pred.argmax(1) == y.argmax(1)).float().mean().item()
      loss, current = loss.item(), (batch + 1) * len(X)

      print(f"epoch {epoch} loss: {loss:>7f}  [{current:>5d}/{len(word_dataset):>5d}] accuracy: {accuracy:>7f}  [{current:>5d}/{len(word_dataset):>5d}]")
      