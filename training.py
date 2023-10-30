import models
import torch
import data
def trainClassifierOneEpoch(
      model: models.ClassifierModule, dataloader: data.ClassDataLoader,
      epoch_index: int = 0, *, optimizer = None, loss_fn = torch.nn.BCELoss()):
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001) if optimizer is None else optimizer
  sum_loss = 0
  corrects = 0
  for data in dataloader:
      input_ids, attn_mask, labels = data
      outputs = model(input_ids, attn_mask)

      print(outputs)
      print(labels)

      pred = torch.argmax(outputs, dim=1)
      corrects += torch.sum(labels[pred] == 1).item()

      optimizer.zero_grad()

      loss = loss_fn(outputs, labels)
      sum_loss += loss.item()
      print(loss)
      loss.backward()

      optimizer.step()

  mean_loss = sum_loss / len(dataloader)
  return mean_loss