import numpy as np
import json
#model
class Model:
    def __init__(self,input_size=2,hidden1_size=5,hidden2_size=5,out_size=1,learning_rate=0.001,activation: str="sigmoid"):
        #implemnt the model
        self.input_size=input_size
        self.hidden1_size=hidden1_size
        self.hidden2_size=hidden2_size
        self.output_size=out_size
        #weights
        self.weight1=np.random.rand(self.input_size,self.hidden1_size)
        self.weight2=np.random.rand(self.hidden1_size,self.hidden2_size)
        self.weight3=np.random.rand(self.hidden2_size,self.output_size)
        #每個neural都應該要有一個bias
        self.bias1=0.01
        self.bias2=0.01
        self.bias3=0.01

        self.learning_rate=learning_rate
        self.activation=activation
    #activation function
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    def sigmoid_derivative(self, x):
        return x*(1-x)
    def relu(self,x):
        return np.maximum(0,x)
    def relu_derivative(self,x):
        return (x>0).astype(float)
    def none(self,x):
        return x
    def none_derivative(self, x):
        return np.ones_like(x)
    def forward(self,x):
        #y=wx+b
        if self.activation=="sigmoid":
            act=self.sigmoid
        elif self.activation=="none":
            act=self.none
        elif self.activation=="relu":
            act=self.relu
        self.hidden1_output=act(np.dot(x,self.weight1)+self.bias1)
        self.hidden2_output=act(np.dot(self.hidden1_output,self.weight2)+self.bias2)
        self.y=act(np.dot(self.hidden2_output,self.weight3)+self.bias3)
        return self.y
    def backward(self,x,y_gt):
        if self.activation=="sigmoid":
            act=self.sigmoid_derivative
        elif self.activation=="none":
            act=self.none_derivative
        elif self.activation=="relu":
            act=self.relu_derivative
        #gt is the ground truth
        y_error=self.y-y_gt
        y_delta=y_error*act(self.y)
        #update weights
        self.weight3-=self.learning_rate*np.dot(self.hidden2_output.T,y_delta)
        self.bias3-=self.learning_rate*y_delta

        #hidden2 error
        hidden2_error=np.dot(y_delta,self.weight3.T)
        hidden2_delta=hidden2_error*act(self.hidden2_output)
        #update weights level2
        self.weight2-=self.learning_rate*np.dot(self.hidden1_output.T,hidden2_delta)
        self.bias2-=self.learning_rate*hidden2_delta

        #hidden1 error
        hidden1_error=np.dot(hidden2_delta,self.weight2.T)
        hidden1_delta=hidden1_error*act(self.hidden1_output)
        #update weights level1
        self.weight1-=self.learning_rate*np.dot(x.T,hidden1_delta)
        self.bias1-=self.learning_rate*hidden1_delta
        return (1/2)*np.sum((self.y-y_gt)**2)
    