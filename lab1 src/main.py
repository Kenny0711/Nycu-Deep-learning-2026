import numpy as np
import matplotlib.pyplot as plt
import json
from data import gernerate_linear, generate_XOR_easy
from model import Model
#show result
def show_result(x, y, pred_y):
    import matplotlib.pyplot as plt
    plt.subplot(1, 2, 1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.subplot(1, 2, 2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.show()
#load JSON
with open('config.json','r') as f :
    config = json.load(f)
print("config:",config)
if config["data"]=="linear":
    x, y = gernerate_linear(n=100)
elif config["data"]=="XOR":
    x, y = generate_XOR_easy()

epochs = config["epoch"]
model = Model(
    input_size=config["input_size"],
    hidden1_size=config["hidden1_size"],
    hidden2_size=config["hidden2_size"],
    out_size=config["output_size"],
    learning_rate=config["learning_rate"]
)
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
plt.show()
# testing
print("########## Testing ##########")
y_pred_probability = model.forward(x)
print(y_pred_probability)
y_pred = np.where(y_pred_probability > 0.5, 1, 0)
print("accuracy: ", sum(1 for i, j in zip(y, y_pred) if i == j) / len(y))
show_result(x, y, y_pred)