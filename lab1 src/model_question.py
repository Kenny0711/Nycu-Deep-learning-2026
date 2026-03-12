import numpy as np
import json
#model
class Model:
    def __init__(self,input_size=2,hidden1_size=5,hidden2_size=5,out_size=1
                 ,learning_rate=0.001,activation: str="sigmoid",optimizer: str="no"):
        #implemnt the model
        self.input_size=input_size
        self.hidden1_size=hidden1_size
        self.hidden2_size=hidden2_size
        self.output_size=out_size
        #weights
        np.random.seed(42)
        self.weight1=np.random.randn(self.input_size+1,self.hidden1_size)
        self.weight2=np.random.randn(self.hidden1_size+1,self.hidden2_size)
        self.weight3=np.random.randn(self.hidden2_size+1,self.output_size)
        #momentum
        self.beta=0.9
        self.v_weight1=np.zeros_like(self.weight1)
        self.v_weight2=np.zeros_like(self.weight2)
        self.v_weight3=np.zeros_like(self.weight3)

        self.learning_rate=learning_rate
        self.activation=activation
        self.optimizer=optimizer
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

        self.x_bias=np.append(x,np.ones((x.shape[0],1)),axis=1)
        self.hidden1_output=act(np.dot(self.x_bias,self.weight1))

        self.hidden1_bias=np.append(self.hidden1_output,np.ones((self.hidden1_output.shape[0],1)),axis=1)
        self.hidden2_output=act(np.dot(self.hidden1_bias,self.weight2))

        self.hidden2_bias=np.append(self.hidden2_output,np.ones((self.hidden2_output.shape[0],1)),axis=1)
        self.y=self.sigmoid(np.dot(self.hidden2_bias,self.weight3))
        return self.y
    def backward(self,y_gt):
        if self.activation=="sigmoid":
            act=self.sigmoid_derivative
        elif self.activation=="none":
            act=self.none_derivative
        elif self.activation=="relu":
            act=self.relu_derivative
        weight3_no_bias=self.weight3[:-1, :].copy()
        weight2_no_bias=self.weight2[:-1, :].copy()
        #gt is the ground truth
        y_error=self.y-y_gt
        self.y_delta=y_error*self.sigmoid_derivative(self.y)

        #hidden2 error
        hidden2_error=np.dot(self.y_delta,weight3_no_bias.T)
        self.hidden2_delta=hidden2_error*act(self.hidden2_output)

        #hidden1 error
        hidden1_error=np.dot(self.hidden2_delta,weight2_no_bias.T)
        self.hidden1_delta=hidden1_error*act(self.hidden1_output)
        return (1/2)*np.sum((self.y-y_gt)**2)
    def update(self):
        if self.optimizer=="momentum":
            self.v_weight3=self.v_weight3*self.beta-self.learning_rate*np.dot(self.hidden2_bias.T,self.y_delta)
            self.weight3+=self.v_weight3
            self.v_weight2=self.v_weight2*self.beta-self.learning_rate*np.dot(self.hidden1_bias.T,self.hidden2_delta)
            self.weight2+=self.v_weight2
            self.v_weight1=self.v_weight1*self.beta-self.learning_rate*np.dot(self.x_bias.T,self.hidden1_delta)
            self.weight1+=self.v_weight1
        else :
            self.weight3-=self.learning_rate*np.dot(self.hidden2_bias.T,self.y_delta)
            self.weight2-=self.learning_rate*np.dot(self.hidden1_bias.T,self.hidden2_delta)
            self.weight1-=self.learning_rate*np.dot(self.x_bias.T,self.hidden1_delta)