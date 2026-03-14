import numpy as np
import matplotlib.pyplot as plt
import json
from data import generate_linear, generate_XOR_easy
from model_question import Model
#show result
def show_result(x, y, pred_y):
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
def train(model,x,y,epochs):
    print("training--------------------------")
    loss_list = []
    epoch_list = []

    for epoch in range(epochs):
        model.forward(x)
        loss = model.backpropagation(y)
        model.update()
        if(epoch % 10000 == 0):
            print(f"epoch: {epoch} loss: {loss}")
        if(epoch % 100 == 0):
            loss_list.append(loss)
            epoch_list.append(epoch)
    return epoch_list,loss_list
def test(model,x,y):
    # testing
    print("testing--------------------------")
    y_pred = model.forward(x)
    loss=np.mean((y_pred-y)**2)
    predictions = np.round(y_pred)
    accuracy = np.mean(predictions == y) * 100
    for i in range(len(y)):
        gt = float(y[i][0])
        pred = float(y_pred[i][0])
        print(f"Iter{i:<2} |        Ground truth: {gt:.1f} |        prediction: {pred:.5f} |")
    print(f"loss={loss:.5f} accuracy={accuracy:.2f}%")
    return predictions
def main():
    #load JSON
    with open('config.json','r') as f :
        config = json.load(f)
    print("config:",config)
    if config["data"]=="linear":
        x, y = generate_linear(n=100)
    elif config["data"]=="XOR":
        x, y = generate_XOR_easy()

    model = Model(
        input_size=config["input_size"],
        hidden1_size=config["hidden1_size"],
        hidden2_size=config["hidden2_size"],
        out_size=config["output_size"],
        learning_rate=config["learning_rate"],
        activation=config["activation"],
        optimizer=config["optimizer"]
    )

    epochs = config["epoch"]
    epoch_list,loss_list=train(model,x,y,epochs)
    pred=test(model,x,y)
    plt.title("Learning Curve")
    plt.plot(epoch_list, loss_list)
    plt.ylabel('loss', fontsize = 14)
    plt.xlabel('epochs', fontsize = 14)
    plt.show()
    show_result(x,y,pred)
if __name__=='__main__':
    main()