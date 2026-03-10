import numpy as np
import matplotlib.pyplot as plt
import json
from data import gernerate_linear, generate_XOR_easy
from model import Model
#load JSON
with open('config.json','r') as f :
    config = json.load(f)
print("config:",config)

if config["data"]=="linear":
    x, y = gernerate_linear(n=100)
else:
    x, y = generate_XOR_easy()

epochs = config["epoch"]
model = Model(
    input_size=config["input_size"],
    hidden1_size=config["hidden1_size"],
    hidden2_size=config["hidden2_size"],
    out_size=config["output_size"]
)
# training
print("########## Training ##########")
loss_list = []
epoch_list = []

plt.title("Learning Curve")
for epoch in range(epochs):
    if config["activation"]=="sigmoid":
        y_test = model.forward(x)
        loss = model.backward(x, y)
    else:
        y_test = model.forward_no(x)
        loss = model.backward_no(x, y)
    if(epoch % 5000 == 0):
        print(f"epoch: {epoch} loss: {loss}")
    if(epoch % 100 == 0):
        loss_list.append(loss)
        epoch_list.append(epoch)
plt.plot(epoch_list, loss_list)
plt.ylabel('loss', fontsize = 14)
plt.xlabel('epochs', fontsize = 14)
plt.show()
# testing
print("########## Testing ##########")
if config["activation"]=="sigmoid":
    y_pred_probability = model.forward(x)
    print(y_pred_probability)
else :
    y_pred_probability = model.forward_no(x)
    print(y_pred_probability)
y_pred = np.where(y_pred_probability > 0.5, 1, 0)
print("accuracy: ", sum(1 for i, j in zip(y, y_pred) if i == j) / len(y))