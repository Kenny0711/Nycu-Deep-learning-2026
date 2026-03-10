import numpy as np
import matplotlib.pyplot as plt

from data import gernerate_linear, generate_XOR_easy
from model import Model
x, y = gernerate_linear(n=100)

epochs = 100001
model = Model()
# training
print("########## Training ##########")
loss_list = []
epoch_list = []

plt.title("Learning Curve")
for epoch in range(epochs):
    y_test = model.forward(x)
    loss = model.backward(x, y)
    if(epoch % 5000 == 0):
        print(f"epoch: {epoch} loss: {loss}")
    if(epoch % 100 == 0):
        loss_list.append(loss)
        epoch_list.append(epoch)
plt.plot(epoch_list, loss_list)
plt.ylabel('loss', fontsize = 14)
plt.xlabel('epochs', fontsize = 14)

# testing
print("########## Testing ##########")
y_pred_probability = model.forward(x)
print(y_pred_probability)
y_pred = np.where(y_pred_probability > 0.5, 1, 0)
print("accuracy: ", sum(1 for i, j in zip(y, y_pred) if i == j) / len(y))
