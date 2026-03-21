import numpy as np


class Model:
    def __init__(
        self,
        input_size=2,
        hidden1_size=5,
        hidden2_size=5,
        out_size=1,
        learning_rate=0.001,
        activation: str = "sigmoid",
        optimizer: str = "no",
        conv_filters: int = 2,
        conv_kernel_size: int = 2,
    ):
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = out_size
        self.learning_rate = learning_rate
        self.activation = activation
        self.optimizer = optimizer

        if conv_kernel_size <= 0:
            raise ValueError("conv_kernel_size must be > 0")
        if input_size < conv_kernel_size:
            raise ValueError("input_size must be >= conv_kernel_size")

        self.conv_filters = conv_filters
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_len = input_size - conv_kernel_size + 1
        self.conv_flat_size = self.conv_filters * self.conv_out_len

        np.random.seed(42)
        self.conv_kernel = np.random.randn(self.conv_filters, self.conv_kernel_size) * 0.1
        self.conv_bias = np.zeros((self.conv_filters,))

        self.weight1 = np.random.randn(self.conv_flat_size + 1, self.hidden1_size) * 0.1
        self.weight2 = np.random.randn(self.hidden1_size + 1, self.hidden2_size) * 0.1
        self.weight3 = np.random.randn(self.hidden2_size + 1, self.output_size) * 0.1

        self.beta = 0.9
        self.v_conv_kernel = np.zeros_like(self.conv_kernel)
        self.v_conv_bias = np.zeros_like(self.conv_bias)
        self.v_weight1 = np.zeros_like(self.weight1)
        self.v_weight2 = np.zeros_like(self.weight2)
        self.v_weight3 = np.zeros_like(self.weight3)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def none(self, x):
        return x

    def none_derivative(self, x):
        return np.ones_like(x)

    def _get_activation(self):
        if self.activation == "sigmoid":
            return self.sigmoid, self.sigmoid_derivative
        if self.activation == "none":
            return self.none, self.none_derivative
        if self.activation == "relu":
            return self.relu, self.relu_derivative
        raise ValueError("activation must be 'sigmoid', 'relu', or 'none'")

    def _conv1d_forward(self, x):
        n = x.shape[0]
        conv_raw = np.zeros((n, self.conv_filters, self.conv_out_len))
        for f in range(self.conv_filters):
            for i in range(self.conv_out_len):
                x_slice = x[:, i : i + self.conv_kernel_size]
                conv_raw[:, f, i] = np.sum(x_slice * self.conv_kernel[f], axis=1) + self.conv_bias[f]
        return conv_raw

    def forward(self, x):
        act, _ = self._get_activation()

        self.x_input = x
        self.conv_raw = self._conv1d_forward(x)
        self.conv_output = act(self.conv_raw)

        self.conv_flat = self.conv_output.reshape(x.shape[0], -1)
        self.conv_bias_input = np.append(self.conv_flat, np.ones((self.conv_flat.shape[0], 1)), axis=1)

        self.hidden1_output = act(np.dot(self.conv_bias_input, self.weight1))
        self.hidden1_bias = np.append(self.hidden1_output, np.ones((self.hidden1_output.shape[0], 1)), axis=1)

        self.hidden2_output = act(np.dot(self.hidden1_bias, self.weight2))
        self.hidden2_bias = np.append(self.hidden2_output, np.ones((self.hidden2_output.shape[0], 1)), axis=1)

        self.y = act(np.dot(self.hidden2_bias, self.weight3))
        return self.y

    def backpropagation(self, y_gt):
        _, act_derivative = self._get_activation()
        batch = y_gt.shape[0]

        weight3_no_bias = self.weight3[:-1, :].copy()
        weight2_no_bias = self.weight2[:-1, :].copy()
        weight1_no_bias = self.weight1[:-1, :].copy()

        y_error = self.y - y_gt
        self.y_delta = (2.0 / batch) * y_error * act_derivative(self.y)

        hidden2_error = np.dot(self.y_delta, weight3_no_bias.T)
        self.hidden2_delta = hidden2_error * act_derivative(self.hidden2_output)

        hidden1_error = np.dot(self.hidden2_delta, weight2_no_bias.T)
        self.hidden1_delta = hidden1_error * act_derivative(self.hidden1_output)

        conv_flat_error = np.dot(self.hidden1_delta, weight1_no_bias.T)
        conv_delta = conv_flat_error.reshape(batch, self.conv_filters, self.conv_out_len)
        conv_delta *= act_derivative(self.conv_output)

        self.grad_conv_kernel = np.zeros_like(self.conv_kernel)
        self.grad_conv_bias = np.zeros_like(self.conv_bias)
        for f in range(self.conv_filters):
            self.grad_conv_bias[f] = np.sum(conv_delta[:, f, :])
            for k in range(self.conv_kernel_size):
                x_slice = self.x_input[:, k : k + self.conv_out_len]
                self.grad_conv_kernel[f, k] = np.sum(conv_delta[:, f, :] * x_slice)

        return np.mean((self.y - y_gt) ** 2)

    def update(self):
        grad_w3 = np.dot(self.hidden2_bias.T, self.y_delta)
        grad_w2 = np.dot(self.hidden1_bias.T, self.hidden2_delta)
        grad_w1 = np.dot(self.conv_bias_input.T, self.hidden1_delta)

        if self.optimizer == "momentum":
            self.v_weight3 = self.v_weight3 * self.beta - self.learning_rate * grad_w3
            self.v_weight2 = self.v_weight2 * self.beta - self.learning_rate * grad_w2
            self.v_weight1 = self.v_weight1 * self.beta - self.learning_rate * grad_w1
            self.v_conv_kernel = self.v_conv_kernel * self.beta - self.learning_rate * self.grad_conv_kernel
            self.v_conv_bias = self.v_conv_bias * self.beta - self.learning_rate * self.grad_conv_bias

            self.weight3 += self.v_weight3
            self.weight2 += self.v_weight2
            self.weight1 += self.v_weight1
            self.conv_kernel += self.v_conv_kernel
            self.conv_bias += self.v_conv_bias
        else:
            self.weight3 -= self.learning_rate * grad_w3
            self.weight2 -= self.learning_rate * grad_w2
            self.weight1 -= self.learning_rate * grad_w1
            self.conv_kernel -= self.learning_rate * self.grad_conv_kernel
            self.conv_bias -= self.learning_rate * self.grad_conv_bias
