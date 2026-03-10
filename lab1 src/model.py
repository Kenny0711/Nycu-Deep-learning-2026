import numpy as np
#model
class Model:
    def __init__(self):
        #implemnt the model
        self.input_size=2
        self.hidden1_size=5
        self.hidden2_size=5
        self.output_size=1
        #weights
        self.weight1=np.random.rand(self.input_size,self.hidden1_size)
        self.weight2=np.random.rand(self.hidden1_size,self.hidden2_size)
        self.weight3=np.random.rand(self.hidden2_size,self.output_size)

        self.bias1=0.001
        self.bias2=0.001
        self.bias3=0.001

        self.learning_rate=0.001

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    def sigmoid_derivative(self, x):
        return x*(1-x)

    def forward(self, x):
        self.hidden1_output=self.sigmoid(np.dot(x,self.weight1)+self.bias1)
        self.hidden2_output=self.sigmoid(np.dot(self.hidden1_output,self.weight2)+self.bias2)
        self.y=self.sigmoid(np.dot(self.hidden2_output,self.weight3)+self.bias3)
        return self.y
    def backward(self,x,y_gt):
        #gt is the ground truth
        y_error=self.y-y_gt
        y_delta=y_error*self.sigmoid_derivative(self.y)
        #update weights
        self.weight3-=self.learning_rate*np.dot(self.hidden2_output.T,y_delta)
        self.bias3-=self.learning_rate*y_delta

        #hidden2 error
        hidden2_error=np.dot(y_delta,self.weight3.T)
        hidden2_delta=hidden2_error*self.sigmoid_derivative(self.hidden2_output)
        #update weights level2
        self.weight2-=self.learning_rate*np.dot(self.hidden1_output.T,hidden2_delta)
        self.bias2-=self.learning_rate*hidden2_delta

        #hidden1 error
        hidden1_error=np.dot(hidden2_delta,self.weight2.T)
        hidden1_delta=hidden1_error*self.sigmoid_derivative(self.hidden1_output)
        #update weights level1
        self.weight1-=self.learning_rate*np.dot(x.T,hidden1_delta)
        self.bias1-=self.learning_rate*hidden1_delta

        return (1/2)*np.sum((self.y-y_gt)**2)
    #不使用sigmoid
    def forward_no(self,x):
      self.hidden1_output=np.dot(x,self.weight1)+self.bias1
      self.hidden2_output=np.dot(self.hidden1_output,self.weight2)+self.bias2
      self.y=np.dot(self.hidden2_output,self.weight3)+self.bias3
      return self.y
    def backward_no(self,x,y_gt):
      y_error=self.y-y_gt
      y_delta=y_error
      self.weight3-=self.learning_rate*np.dot(self.hidden2_output.T,y_delta)
      self.bias3-=self.learning_rate*y_delta
      hidden2_error=np.dot(y_delta,self.weight3.T)
      hidden2_delta=hidden2_error
      self.weight2-=self.learning_rate*np.dot(self.hidden1_output.T,hidden2_delta)
      self.bias2-=self.learning_rate*hidden2_delta
      hidden1_error=np.dot(hidden2_delta,self.weight2.T)
      hidden1_delta=hidden1_error
      self.weight1-=self.learning_rate*np.dot(x.T,hidden1_delta)
      self.bias1-=self.learning_rate*hidden1_delta
      return (1/2)*np.sum((self.y-y_gt)**2)
    
