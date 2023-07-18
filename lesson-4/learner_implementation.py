# -*- coding: utf-8 -*-
"""Learner Implementation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ffVlN5sgnPyy_EkhZYy07i5rhqFOtG7Q
"""

import fastbook
fastbook.setup_book()

from fastai.vision.all import *
from fastbook import *


"""# Assemble Dataset

For this initial Learner we are just going to try to create a model that can classify any image as a 3 or a 7. So let's download a sample of MNIST that contains images of just these digits:

"""

path = untar_data(URLs.MNIST_SAMPLE)
Path.BASE_PATH = path
training_threes = (path/'train'/'3').ls().sorted()
training_sevens = (path/'train'/'7').ls().sorted()
Image.open(training_sevens[0])

three_tensors = [tensor(Image.open(o)) for o in training_threes]
seven_tensors = [tensor(Image.open(o)) for o in training_sevens]
len(three_tensors),len(seven_tensors)

stacked_threes = torch.stack(three_tensors).float()/255
stacked_sevens = torch.stack(seven_tensors).float()/255
stacked_threes.shape

train_x = torch.cat([stacked_threes, stacked_sevens]).view(-1, 28*28)
train_x.shape

# If the image is a 3 then the validation set will have a 1,
# If the image is a 7 then the validation set will have a 0
train_y = tensor([1]*len(training_threes) + [0]*len(training_sevens)).unsqueeze(1)

dataset = list(zip(train_x, train_y))
x, y = dataset[0]
x.shape, y

"""# Leaner"""

import random

class MyDataLoader:

  # Dataset should be a tuple of tensors representing batches of independent and dependent variables
  def __init__(self, dataset, batch_size):
    self.dataset = dataset
    self.batch_size = batch_size

  def get_batch(self):
    random.shuffle(self.dataset)
    return self.dataset[:self.batch_size]

# Params = weights + bias together in one matrix
class MyBasicOptimizer:
    def __init__(self, params, learning_rate):
      self.params = params
      self.learning_rate = learning_rate

    def step(self, *args, **kwargs):
        for p in self.params:
          p.data -= p.grad.data * self.learning_rate

    def zero_grad(self, *args, **kwargs):
        for p in self.params:
          p.grad.zero_()

class MyModel:
  def __init__(self, size):
    self.weights1 = self._init_params((28*28, size))
    self.bias1 = self._init_params(size)

  def _init_params(self, size, std=1.0):
    return (torch.randn(size)*std).requires_grad_()

  # Simple net
  def predict(self, inputs):
      res = inputs @ self.weights1 + self.bias1
      return res

  def parameters(self):
    params = [self.weights1, self.bias1]
    return params

def mnist_loss(predictions, targets):
  # predictions = predictions.sigmoid()
  return torch.where(targets==1, 1-predictions, predictions).mean()

def batch_accuracy(xb, yb):
  with torch.no_grad():
    preds = xb
    correct = (preds>0.5) == yb
    return correct.float().mean()

dataloader = MyDataLoader(dataset, batch_size=512)

model = MyModel(1)

optimizer = MyBasicOptimizer(model.parameters(), 0.1)

class MyLearner:
  def __init__(self, dataloader, model, optimizer, loss_func, metrics):
    self.dataloader = dataloader
    self.model = model
    self.optimizer = optimizer
    self.loss_func = loss_func
    self.metrics = metrics

  def _apply_step(self, should_print=True):
      dataset = self.dataloader.get_batch()
      inputs = torch.stack([x[0] for x in dataset])
      targets = torch.stack([x[1] for x in dataset])

      predictions = self.model.predict(inputs)
      loss = self.loss_func(predictions, targets)
      loss.backward()

      self.optimizer.step()
      self.optimizer.zero_grad()

      if should_print:
        print(loss.item())
        print(self.metrics(predictions, targets))


  def fit(self, iterations):
    for i in range(iterations):
      self._apply_step()

learn = MyLearner(dataloader, model, optimizer, mnist_loss, batch_accuracy)

learn.fit(100)

# Validation

validation_threes = (path/'valid'/'3').ls().sorted()
top_valid_threes = validation_threes[:20]
three_tensors = [tensor(Image.open(o)) for o in top_valid_threes]
stacked_valid_threes = torch.stack(three_tensors).float()/255
valid_x = stacked_valid_threes.view(-1, 28*28)
prediction = model.predict(valid_x).sigmoid()
print(prediction)
print(prediction.shape)
print(batch_accuracy(prediction, tensor([1]*len(top_valid_threes)).unsqueeze(1)))


validation_sevens = (path/'valid'/'7').ls().sorted()
top_valid_sevens = validation_sevens[:20]
seven_tensors = [tensor(Image.open(o)) for o in top_valid_sevens]
stacked_valid_sevens = torch.stack(seven_tensors).float()/255
valid_x = stacked_valid_sevens.view(-1, 28*28)
prediction = model.predict(valid_x).sigmoid()
print(prediction)
print(prediction.shape)
print(batch_accuracy(prediction, tensor([0]*len(top_valid_threes)).unsqueeze(1)))