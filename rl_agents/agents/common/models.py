import numpy as np
import torch
import torch.nn as nn
from gym import spaces
from torch.nn import functional as F
import copy
from rl_agents.configuration import Configurable


class BaseModule(torch.nn.Module):
    """
        Base torch.nn.Module implementing basic features:
            - initialization factory
            - normalization parameters
    """

    def __init__(self, activation_type="RELU", reset_type="XAVIER", normalize=None):
        super().__init__()
        self.activation = activation_factory(activation_type)
        self.reset_type = reset_type
        self.normalize = normalize
        self.mean = None
        self.std = None

    def _init_weights(self, m):
        if hasattr(m, 'weight'):
            if self.reset_type == "XAVIER":
                torch.nn.init.xavier_uniform_(m.weight.data)
            elif self.reset_type == "ZEROS":
                torch.nn.init.constant_(m.weight.data, 0.)
            else:
                raise ValueError("Unknown reset type")
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.)

    def set_normalization_params(self, mean, std):
        if self.normalize:
            std[std == 0.] = 1.
        self.std = std
        self.mean = mean

    def reset(self):
        self.apply(self._init_weights)

    def forward(self, *input):
        if self.normalize:
            input = (input.float() - self.mean.float()) / self.std.float()
        return NotImplementedError


class MultiLayerPerceptron(BaseModule, Configurable):
    def __init__(self, config):
        super().__init__()
        Configurable.__init__(self, config)
        sizes = [self.config["in"]] + self.config["layers"]
        self.activation = activation_factory(self.config["activation"])
        layers_list = [nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        self.layers = nn.ModuleList(layers_list)
        if self.config.get("out", None):
            self.predict = nn.Linear(sizes[-1], self.config["out"])

    @classmethod
    def default_config(cls):
        return {"in": None,
                "layers": [64, 64],
                "activation": "RELU",
                "reshape": "True",
                "out": None}

    def forward(self, x):
        if self.config["reshape"]:
            x = x.reshape(x.shape[0], -1)  # We expect a batch of vectors
        for layer in self.layers:
            x = self.activation(layer(x))
        if self.config.get("out", None):
            x = self.predict(x)
        return x


class DuelingNetwork(BaseModule, Configurable):
    def __init__(self, config):
        super().__init__()
        Configurable.__init__(self, config)
        self.config["base_module"]["in"] = self.config["in"]
        self.base_module = model_factory(self.config["base_module"])
        self.config["value"]["in"] = self.base_module.config["layers"][-1]
        self.config["value"]["out"] = 1
        self.value = model_factory(self.config["value"])
        self.config["advantage"]["in"] = self.base_module.config["layers"][-1]
        self.config["advantage"]["out"] = self.config["out"]
        self.advantage = model_factory(self.config["advantage"])

    @classmethod
    def default_config(cls):
        return {"in": None,
                "base_module": {"type": "MultiLayerPerceptron", "out": None},
                "value": {"type": "MultiLayerPerceptron", "layers": [], "out": None},
                "advantage": {"type": "MultiLayerPerceptron", "layers": [], "out": None},
                "out": None}

    def forward(self, x):
        x = self.base_module(x)
        value = self.value(x).expand(-1, self.config["out"])
        advantage = self.advantage(x)
        return value + advantage - advantage.mean(1).unsqueeze(1).expand(-1, self.config["out"])

# we need a convolution layer and since PyTorch does not have the 'auto' padding in Conv2d, so we have to code ourself!
class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2) # dynamic add padding based on the kernel_size

class ConvNetAtari(nn.Module, Configurable):
    def __init__(self, config):
        super().__init__()
        Configurable.__init__(self, config)
        self.activation = activation_factory(self.config["activation"])
        self.conv1 = nn.Conv2d(self.config["in_channels"], 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)

        # MLP Head
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(self.config["in_width"], kernel_size=8, stride=4), kernel_size=4, stride=2)
        convh = conv2d_size_out(conv2d_size_out(self.config["in_height"], kernel_size=8, stride=4), kernel_size=4, stride=2)
        assert convh > 0 and convw > 0
        self.config["head_mlp"]["in"] = convw * convh * 32
        self.config["head_mlp"]["out"] = self.config["out"]

        self.fc1 = nn.Linear(self.config["head_mlp"]["in"], 256)
        self.fc2 = nn.Linear(256, self.config["head_mlp"]["out"])

    @classmethod
    def default_config(cls):
        return {
            "in_channels": None,
            "in_height": None,
            "in_width": None,
            "activation": "RELU",
            "head_mlp": {
                "type": "MultiLayerPerceptron",
                "in": None,
                "layers": [],
                "activation": "RELU",
                "reshape": "True",
                "out": None
            },
            "out": None
        }

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        """
            Forward convolutional network
        :param x: tensor of shape BCHW
        """
        x = self.activation((self.conv1(x)))
        x = self.activation((self.conv2(x)))

        # x = x.view(-1, self.num_flat_features(x))
        x = torch.flatten(x,1)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)

        return x
class ConvNetAtariDoubleQ(nn.Module, Configurable):
    def __init__(self, config):
        super().__init__()
        Configurable.__init__(self, config)
        self.activation = activation_factory(self.config["activation"])
        self.conv1 = nn.Conv2d(self.config["in_channels"], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # MLP Head
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.config["in_width"], kernel_size=8, stride=4), kernel_size=4, stride=2))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.config["in_height"], kernel_size=8, stride=4), kernel_size=4, stride=2))
        assert convh > 0 and convw > 0
        self.config["head_mlp"]["in"] = convw * convh * 64
        self.config["head_mlp"]["out"] = self.config["out"]

        self.fc1 = nn.Linear(self.config["head_mlp"]["in"], 512)
        self.fc2 = nn.Linear(512, self.config["head_mlp"]["out"])

    @classmethod
    def default_config(cls):
        return {
            "in_channels": None,
            "in_height": None,
            "in_width": None,
            "activation": "RELU",
            "head_mlp": {
                "type": "MultiLayerPerceptron",
                "in": None,
                "layers": [],
                "activation": "RELU",
                "reshape": "True",
                "out": None
            },
            "out": None
        }

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        """
            Forward convolutional network
        :param x: tensor of shape BCHW
        """
        x = self.activation((self.conv1(x)))
        x = self.activation((self.conv2(x)))
        x = self.activation((self.conv3(x)))
        x = torch.flatten(x, 1)
        # x = x.view(-1, self.num_flat_features(x))
        x = self.activation(self.fc1(x))
        x = self.fc2(x)

        return x


class ConvNet3Layer(nn.Module, Configurable):
    def __init__(self, config):
        super().__init__()
        Configurable.__init__(self, config)
        self.activation = activation_factory(self.config["activation"])
        self.conv1 = nn.Conv2d(self.config["in_channels"], 16, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=2, stride=2)

        # MLP Head
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=2, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.config["in_width"])))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.config["in_height"])))
        assert convh > 0 and convw > 0
        self.config["head_mlp"]["in"] = convw * convh * 64
        self.config["head_mlp"]["out"] = self.config["out"]
        self.head = model_factory(self.config["head_mlp"])

    @classmethod
    def default_config(cls):
        return {
            "in_channels": None,
            "in_height": None,
            "in_width": None,
            "activation": "RELU",
            "head_mlp": {
                "type": "MultiLayerPerceptron",
                "in": None,
                "layers": [],
                "activation": "RELU",
                "reshape": "True",
                "out": None
            },
            "out": None
        }

    def forward(self, x):
        """
            Forward convolutional network
        :param x: tensor of shape BCHW
        """
        x = self.activation((self.conv1(x)))
        x = self.activation((self.conv2(x)))
        x = self.activation((self.conv3(x)))
        return self.head(x)
class ConvNet3LayerVariableKernel(nn.Module, Configurable):
    def __init__(self, config):
        super().__init__()
        Configurable.__init__(self, config)
        self.activation = activation_factory(self.config["activation"])
        self.conv1 = nn.Conv2d(self.config["in_channels"], 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1)

        # MLP Head
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(self.config["in_width"], kernel_size=5, stride=2))))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(self.config["in_height"], kernel_size=5, stride=2))))
        assert convh > 0 and convw > 0
        self.config["head_mlp"]["in"] = convw * convh * 128
        self.config["head_mlp"]["out"] = self.config["out"]
        self.head = model_factory(self.config["head_mlp"])

    @classmethod
    def default_config(cls):
        return {
            "in_channels": None,
            "in_height": None,
            "in_width": None,
            "activation": "RELU",
            "head_mlp": {
                "type": "MultiLayerPerceptron",
                "in": None,
                "layers": [],
                "activation": "RELU",
                "reshape": "True",
                "out": None
            },
            "out": None
        }

    def forward(self, x):
        """
            Forward convolutional network
        :param x: tensor of shape BCHW
        """
        x = self.activation((self.conv1(x)))
        x = self.activation((self.conv2(x)))
        x = self.activation((self.conv3(x)))
        x = self.activation((self.conv4(x)))
        return self.head(x)


class ConvNetStanfordMARLNoRes(nn.Module, Configurable):

    def __init__(self, config):
        super().__init__()
        Configurable.__init__(self, config)
        self.activation = activation_factory(self.config["activation"])
        self.conv1 = nn.Conv2d(self.config["in_channels"], 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.maxpool1 = nn.MaxPool2d(3, stride=2)

        # MLP Head
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        def maxpool_size_out(size, kernel_size=3, stride=2, padding=0):
            return (size + 2 * padding - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(
            conv2d_size_out(
                maxpool_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(self.config["in_width"]))))))
        convh = conv2d_size_out(
            conv2d_size_out(
                maxpool_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(self.config["in_height"]))))))

        assert convh > 0 and convw > 0
        self.config["head_mlp"]["in"] = convw * convh * 32
        self.config["head_mlp"]["out"] = self.config["out"]
        self.head = model_factory(self.config["head_mlp"])

    @classmethod
    def default_config(cls):
        return {
            "in_channels": None,
            "in_height": None,
            "in_width": None,
            "activation": "RELU",
            "head_mlp": {
                "type": "MultiLayerPerceptron",
                "in": None,
                "layers": [],
                "activation": "RELU",
                "reshape": "True",
                "out": None
            },
            "out": None
        }

    def forward(self, x):
        """
            Forward convolutional network
        :param x: tensor of shape BCHW
        """
        x = self.activation((self.conv1(x)))
        x = self.activation((self.conv2(x)))
        x = self.bn1(x)
        x = self.activation((self.conv3(x)))
        x = self.maxpool1(x)
        x = self.activation((self.conv4(x)))
        x = self.bn2(x)
        x = self.activation((self.conv5(x)))
        return self.head(x)
class ConvNetStanfordMARLRes(nn.Module, Configurable):

    def __init__(self, config):
        super().__init__()
        Configurable.__init__(self, config)
        self.activation = activation_factory(self.config["activation"])
        self.conv1 = nn.Conv2d(self.config["in_channels"], 32, kernel_size=3, stride=1, padding =1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(3, stride=2)
        self.bn1 = nn.BatchNorm2d(32)



        # MLP Head
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=3, stride=1,padding =1):
            return (size +2*padding - (kernel_size - 1) - 1) // stride + 1

        def maxpool_size_out(size, kernel_size=3, stride=2, padding=0):
            return (size + 2 * padding - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(
            conv2d_size_out(maxpool_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(self.config["in_width"]))))))
        convh = conv2d_size_out(
            conv2d_size_out(maxpool_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(self.config["in_height"]))))))
        assert convh > 0 and convw > 0
        self.config["head_mlp"]["in"] = convw * convh * 32
        self.config["head_mlp"]["out"] = self.config["out"]
        self.head = model_factory(self.config["head_mlp"])

    @classmethod
    def default_config(cls):
        return {
            "in_channels": None,
            "in_height": None,
            "in_width": None,
            "activation": "RELU",
            "head_mlp": {
                "type": "MultiLayerPerceptron",
                "in": None,
                "layers": [],
                "activation": "RELU",
                "reshape": "True",
                "out": None
            },
            "out": None
        }

    def forward(self, x):
        """
            Forward convolutional network
        :param x: tensor of shape BCHW
        """
        x = self.activation((self.conv1(x)))
        # x_residual1 = x.detach().clone()
        # x_residual1 = torch.tensor(x)
        # x_residual1 = x.detach().clone()
        x_residual1 = x
        x = self.activation((self.conv2(x)))
        x = self.bn1(x)
        x = self.activation((self.conv2(x)))
        x += x_residual1
        x = self.maxpool1(x)
        # x_residual2 = x.detach().clone()
        x_residual2 = x
        x = self.activation((self.conv2(x)))
        x = self.bn1(x)
        x = self.activation((self.conv2(x)))
        x += x_residual2
        return self.head(x)


class ConvNet3D(nn.Module, Configurable):
    def __init__(self, config):
        super().__init__()
        Configurable.__init__(self, config)
        self.activation = activation_factory(self.config["activation"])
        self.conv1 = nn.Conv3d(self.config["in_channels"], 32, kernel_size=(1, 8, 8), stride=(1, 4, 4))
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 4, 4), stride=(1, 2, 2))
        self.conv3 = nn.Conv3d(64, 64,  kernel_size=(3, 3, 3), stride=(1, 1, 1))
        # MLP Head
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.config["in_width"], kernel_size=8, stride=4), kernel_size=4, stride=2))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.config["in_height"], kernel_size=8, stride=4), kernel_size=4, stride=2))
        convd = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.config["in_depth"], kernel_size=3, stride=1), kernel_size=1,stride=1))


        assert convh > 0 and convw > 0
        self.config["head_mlp"]["in"] = convw * convh * convd * 64
        self.config["head_mlp"]["out"] = self.config["out"]

        self.fc1 = nn.Linear(self.config["head_mlp"]["in"], 512)
        self.fc2 = nn.Linear(512, self.config["head_mlp"]["out"])

    @classmethod
    def default_config(cls):
        return {
            "in_channels": None,
            "in_height": None,
            "in_width": None,
            "activation": "RELU",
            "head_mlp": {
                "type": "MultiLayerPerceptron",
                "in": None,
                "layers": [],
                "activation": "RELU",
                "reshape": "True",
                "out": None
            },
            "out": None
        }

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        """
            Forward convolutional network
        :param x: tensor of shape BCHW
        """
        x = self.activation((self.conv1(x)))
        x = self.activation((self.conv2(x)))
        x = self.activation((self.conv3(x)))
        x = torch.flatten(x, 1)
        # x = x.view(-1, self.num_flat_features(x))
        x = self.activation(self.fc1(x))
        x = self.fc2(x)

        return x

class ConvNet3DResidual(nn.Module, Configurable):
    def __init__(self, config):
        super().__init__()
        Configurable.__init__(self, config)
        self.activation = activation_factory(self.config["activation"])
        self.conv1 = nn.Conv3d(self.config["in_channels"], 32, kernel_size=(1, 7, 7), stride=(1, 1, 1), padding =(0,3,3))
        self.conv2 = nn.Conv3d(32, 32, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding = (1,2,2))
        self.conv3 = nn.Conv3d(32, 64,  kernel_size=(3, 5, 5), stride=(1, 1, 1))
        self.conv4 = nn.Conv3d(64, 64, kernel_size=(1, 5, 5), stride=(1, 1, 1))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(3, 5, 5), stride=(1, 2, 2))
        self.bn1 = nn.BatchNorm3d(64)
        # MLP Head
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=1, padding=0):
            return (size + 2 * padding - (kernel_size - 1) - 1) // stride + 1

        def maxpool_size_out(size, kernel_size=5, stride=2, padding=0):
            return (size + 2 * padding - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(maxpool_size_out(conv2d_size_out(conv2d_size_out(self.config["in_width"], kernel_size=7, stride=1,padding=3), kernel_size=5, stride=1,padding=2))))
        convh = conv2d_size_out(conv2d_size_out(maxpool_size_out(conv2d_size_out(conv2d_size_out(self.config["in_height"], kernel_size=7, stride=1,padding=3), kernel_size=5, stride=1,padding=2))))
        convd = conv2d_size_out(conv2d_size_out(maxpool_size_out(conv2d_size_out(conv2d_size_out(self.config["in_depth"], kernel_size=1, stride=1,padding=0), kernel_size=3, stride=1,padding=1),  kernel_size=3, stride=1),kernel_size=3, stride=1),kernel_size=1, stride=1)


        assert convh > 0 and convw > 0
        self.config["head_mlp"]["in"] = convw * convh * convd * 64
        self.config["head_mlp"]["out"] = self.config["out"]

        self.fc1 = nn.Linear(self.config["head_mlp"]["in"], 256)
        self.fc2 = nn.Linear(256, self.config["head_mlp"]["out"])

    @classmethod
    def default_config(cls):
        return {
            "in_channels": None,
            "in_height": None,
            "in_width": None,
            "activation": "RELU",
            "head_mlp": {
                "type": "MultiLayerPerceptron",
                "in": None,
                "layers": [],
                "activation": "RELU",
                "reshape": "True",
                "out": None
            },
            "out": None
        }

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        """
            Forward convolutional network
        :param x: tensor of shape BCHW
        """
        x = self.activation((self.conv1(x)))
        x_residual1 = x
        x = self.activation((self.conv2(x)))
        x += x_residual1
        x = self.maxpool1(x)
        x = self.activation((self.conv3(x)))
        x = self.bn1(x)
        x = self.activation((self.conv4(x)))

        x = torch.flatten(x, 1)
        # x = x.view(-1, self.num_flat_features(x))
        x = self.activation(self.fc1(x))
        x = self.fc2(x)

        return x


class EgoAttention(BaseModule, Configurable):
    def __init__(self, config):
        super().__init__()
        Configurable.__init__(self, config)
        self.features_per_head = int(self.config["feature_size"] / self.config["heads"])

        self.value_all = nn.Linear(self.config["feature_size"], self.config["feature_size"], bias=False)
        self.key_all = nn.Linear(self.config["feature_size"], self.config["feature_size"], bias=False)
        self.query_ego = nn.Linear(self.config["feature_size"], self.config["feature_size"], bias=False)
        self.attention_combine = nn.Linear(self.config["feature_size"], self.config["feature_size"], bias=False)

    @classmethod
    def default_config(cls):
        return {
            "feature_size": 64,
            "heads": 4,
            "dropout_factor": 0,
        }

    def forward(self, ego, others, mask=None):
        batch_size = others.shape[0]
        n_entities = others.shape[1] + 1
        input_all = torch.cat((ego.view(batch_size, 1, self.config["feature_size"]), others), dim=1)
        # Dimensions: Batch, entity, head, feature_per_head
        key_all = self.key_all(input_all).view(batch_size, n_entities, self.config["heads"], self.features_per_head)
        value_all = self.value_all(input_all).view(batch_size, n_entities, self.config["heads"], self.features_per_head)
        query_ego = self.query_ego(ego).view(batch_size, 1, self.config["heads"], self.features_per_head)

        # Dimensions: Batch, head, entity, feature_per_head
        key_all = key_all.permute(0, 2, 1, 3)
        value_all = value_all.permute(0, 2, 1, 3)
        query_ego = query_ego.permute(0, 2, 1, 3)
        if mask is not None:
            mask = mask.view((batch_size, 1, 1, n_entities)).repeat((1, self.config["heads"], 1, 1))
        value, attention_matrix = attention(query_ego, key_all, value_all, mask,
                                            nn.Dropout(self.config["dropout_factor"]))
        result = (self.attention_combine(value.reshape((batch_size, self.config["feature_size"]))) + ego.squeeze(1)) / 2
        return result, attention_matrix


class SelfAttention(BaseModule, Configurable):
    def __init__(self, config):
        super().__init__()
        Configurable.__init__(self, config)
        self.features_per_head = int(self.config["feature_size"] / self.config["heads"])

        self.value_all = nn.Linear(self.config["feature_size"], self.config["feature_size"], bias=False)
        self.key_all = nn.Linear(self.config["feature_size"], self.config["feature_size"], bias=False)
        self.query_all = nn.Linear(self.config["feature_size"], self.config["feature_size"], bias=False)
        self.attention_combine = nn.Linear(self.config["feature_size"], self.config["feature_size"], bias=False)

    @classmethod
    def default_config(cls):
        return {
            "feature_size": 64,
            "heads": 4,
            "dropout_factor": 0,
        }

    def forward(self, ego, others, mask=None):
        batch_size = others.shape[0]
        n_entities = others.shape[1] + 1
        input_all = torch.cat((ego.view(batch_size, 1, self.config["feature_size"]), others), dim=1)
        # Dimensions: Batch, entity, head, feature_per_head
        key_all = self.key_all(input_all).view(batch_size, n_entities, self.config["heads"], self.features_per_head)
        value_all = self.value_all(input_all).view(batch_size, n_entities, self.config["heads"], self.features_per_head)
        query_all = self.query_all(input_all).view(batch_size, n_entities, self.config["heads"], self.features_per_head)

        # Dimensions: Batch, head, entity, feature_per_head
        key_all = key_all.permute(0, 2, 1, 3)
        value_all = value_all.permute(0, 2, 1, 3)
        query_all = query_all.permute(0, 2, 1, 3)
        if mask is not None:
            mask = mask.view((batch_size, 1, 1, n_entities)).repeat((1, self.config["heads"], 1, 1))
        value, attention_matrix = attention(query_all, key_all, value_all, mask,
                                            nn.Dropout(self.config["dropout_factor"]))
        result = (self.attention_combine(
            value.reshape((batch_size, n_entities, self.config["feature_size"]))) + input_all) / 2
        return result, attention_matrix


class EgoAttentionNetwork(BaseModule, Configurable):
    def __init__(self, config):
        super().__init__()
        Configurable.__init__(self, config)
        self.config = config
        if not self.config["embedding_layer"]["in"]:
            self.config["embedding_layer"]["in"] = self.config["in"]
        if not self.config["others_embedding_layer"]["in"]:
            self.config["others_embedding_layer"]["in"] = self.config["in"]
        self.config["output_layer"]["in"] = self.config["attention_layer"]["feature_size"]
        self.config["output_layer"]["out"] = self.config["out"]

        self.ego_embedding = model_factory(self.config["embedding_layer"])
        self.others_embedding = model_factory(self.config["others_embedding_layer"])
        self.self_attention_layer = None
        if self.config["self_attention_layer"]:
            self.self_attention_layer = SelfAttention(self.config["self_attention_layer"])
        self.attention_layer = EgoAttention(self.config["attention_layer"])
        self.output_layer = model_factory(self.config["output_layer"])

    @classmethod
    def default_config(cls):
        return {
            "in": None,
            "out": None,
            "presence_feature_idx": 0,
            "embedding_layer": {
                "type": "MultiLayerPerceptron",
                "layers": [128, 128, 128],
                "reshape": False
            },
            "others_embedding_layer": {
                "type": "MultiLayerPerceptron",
                "layers": [128, 128, 128],
                "reshape": False
            },
            "self_attention_layer": {
                "type": "SelfAttention",
                "feature_size": 128,
                "heads": 4
            },
            "attention_layer": {
                "type": "EgoAttention",
                "feature_size": 128,
                "heads": 4
            },
            "output_layer": {
                "type": "MultiLayerPerceptron",
                "layers": [128, 128, 128],
                "reshape": False
            },
        }

    def forward(self, x):
        ego_embedded_att, _ = self.forward_attention(x)
        return self.output_layer(ego_embedded_att)

    def split_input(self, x, mask=None):
        # Dims: batch, entities, features
        ego = x[:, 0:1, :]
        others = x[:, 1:, :]
        if mask is None:
            mask = x[:, :, self.config["presence_feature_idx"]:self.config["presence_feature_idx"] + 1] < 0.5
        return ego, others, mask

    def forward_attention(self, x):
        ego, others, mask = self.split_input(x)
        ego, others = self.ego_embedding(ego), self.others_embedding(others)
        if self.self_attention_layer:
            self_att, _ = self.self_attention_layer(ego, others, mask)
            ego, others, mask = self.split_input(self_att, mask=mask)
        return self.attention_layer(ego, others, mask)

    def get_attention_matrix(self, x):
        _, attention_matrix = self.forward_attention(x)
        return attention_matrix


class AttentionNetwork(BaseModule, Configurable):
    def __init__(self, config):
        super().__init__()
        Configurable.__init__(self, config)
        self.config = config
        if not self.config["embedding_layer"]["in"]:
            self.config["embedding_layer"]["in"] = self.config["in"]
        self.config["output_layer"]["in"] = self.config["attention_layer"]["feature_size"]
        self.config["output_layer"]["out"] = self.config["out"]

        self.embedding = model_factory(self.config["embedding_layer"])
        self.attention_layer = SelfAttention(self.config["attention_layer"])
        self.output_layer = model_factory(self.config["output_layer"])

    @classmethod
    def default_config(cls):
        return {
            "in": None,
            "out": None,
            "presence_feature_idx": 0,
            "embedding_layer": {
                "type": "MultiLayerPerceptron",
                "layers": [128, 128, 128],
                "reshape": False
            },
            "attention_layer": {
                "type": "SelfAttention",
                "feature_size": 128,
                "heads": 4
            },
            "output_layer": {
                "type": "MultiLayerPerceptron",
                "layers": [128, 128, 128],
                "reshape": False
            },
        }

    def forward(self, x):
        ego, others, mask = self.split_input(x)
        ego_embedded_att, _ = self.attention_layer(self.embedding(ego), self.others_embedding(others), mask)
        return self.output_layer(ego_embedded_att)

    def split_input(self, x):
        # Dims: batch, entities, features
        ego = x[:, 0:1, :]
        others = x[:, 1:, :]
        mask = x[:, :, self.config["presence_feature_idx"]:self.config["presence_feature_idx"] + 1] < 0.5
        return ego, others, mask

    def get_attention_matrix(self, x):
        ego, others, mask = self.split_input(x)
        _, attention_matrix = self.attention_layer(self.embedding(ego), self.others_embedding(others), mask)
        return attention_matrix


def attention(query, key, value, mask=None, dropout=None):
    """
        Compute a Scaled Dot Product Attention.
    :param query: size: batch, head, 1 (ego-entity), features
    :param key:  size: batch, head, entities, features
    :param value: size: batch, head, entities, features
    :param mask: size: batch,  head, 1 (absence feature), 1 (ego-entity)
    :param dropout:
    :return: the attention softmax(QK^T/sqrt(dk))V
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    output = torch.matmul(p_attn, value)
    return output, p_attn


def activation_factory(activation_type):
    if activation_type == "RELU":
        return F.relu
    elif activation_type == "TANH":
        return torch.tanh
    else:
        raise ValueError("Unknown activation_type: {}".format(activation_type))


def trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def size_model_config(env, model_config):
    """
        Update the configuration of a model depending on the environment observation/action spaces

        Typically, the input/output sizes.

    :param env: an environment
    :param model_config: a model configuration
    """

    if isinstance(env.observation_space, spaces.Box):
        obs_shape = env.observation_space.shape
    elif isinstance(env.observation_space, spaces.Tuple):
        obs_shape = env.observation_space.spaces[0].shape

    if len(obs_shape) == 3: # Assume CHW observation space
        model_config["in_channels"] = int(obs_shape[0])
        model_config["in_height"] = int(obs_shape[1])
        model_config["in_width"] = int(obs_shape[2])
        model_config["inputc"] = (model_config["in_channels"], model_config["in_height"] ,  model_config["in_width"])

    elif len(obs_shape) == 4: # Assume CDHW observation space conv 3D
        model_config["in_channels"] = int(obs_shape[0])
        model_config["in_depth"] = int(obs_shape[1])
        model_config["in_height"] = int(obs_shape[2])
        model_config["in_width"] = int(obs_shape[3])
        model_config["inputc"] = (model_config["in_channels"],  model_config["in_depth"] ,model_config["in_height"] ,  model_config["in_width"])
    else:
        model_config["in"] = int(np.prod(obs_shape))
        model_config["inputc"] = (int(obs_shape[0]), int(obs_shape[1]))

    if isinstance(env.action_space, spaces.Discrete):
        model_config["out"] = env.action_space.n
    elif isinstance(env.action_space, spaces.Tuple):
        model_config["out"] = env.action_space.spaces[0].n



def model_factory(config: dict) -> nn.Module:
    if config["type"] == "MultiLayerPerceptron":
        return MultiLayerPerceptron(config)
    elif config["type"] == "DuelingNetwork":
        return DuelingNetwork(config)
    elif config["type"] == "ConvNetAtari":
        return ConvNetAtari(config)
    elif config["type"] == "ConvNetAtariDoubleQ":
        return ConvNetAtariDoubleQ(config)
    elif config["type"] == "ConvNet3Layer":
        return ConvNet3Layer(config)
    elif config["type"] == "ConvNet3D":
        return ConvNet3D(config)
    elif config["type"] == "ConvNet3DResidual":
        return ConvNet3DResidual(config)
    elif config["type"] == "ConvNet3LayerVariableKernel":
        return ConvNet3LayerVariableKernel(config)
    elif config["type"] == "ConvNetStanfordMARLNoRes":
        return ConvNetStanfordMARLNoRes(config)
    elif config["type"] == "ConvNetStanfordMARLRes":
        return ConvNetStanfordMARLRes(config)
    elif config["type"] == "EgoAttentionNetwork":
        return EgoAttentionNetwork(config)
    else:
        raise ValueError("Unknown model type")
