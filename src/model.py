import onnx
from onnxruntime import InferenceSession
from skl2onnx import to_onnx
from sklearn.neural_network import MLPRegressor
import numpy as np

import torch
import torch.nn as nn
import torch.onnx


class Model:
    def __init__(self):
        pass

    def predict(self, s):
        raise NotImplementedError()

    def eval_batch(self, obss):
        raise NotImplementedError()

    def eval(self, s):
        raise NotImplementedError()

    def best(self, s):
        raise NotImplementedError()

    def current_loss(self):
        raise NotImplementedError()


class OnnxModel(Model):
    def __init__(self, model):
        super().__init__()
        assert model.has_learned_something

        self.onnx_model, self.session = model.to_onnx()

    def save(self, path):
        onnx.save(self.onnx_model, path + ".onnx")

    def predict(self, s):
        if s is None:
            return 0
        return self.session.run(None, {'X': s})[0]

    def eval_batch(self, ss):
        return np.array([self.eval(s) for s in ss])

    def eval(self, s):
        return np.max(self.predict(s))

    def current_loss(self):
        raise NotImplementedError()


class TorchModel(Model):

    def __init__(self, args, nfeatures, network=None):
        super().__init__()
        self.nfeatures = nfeatures
        self.n, self.k = None, None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using", self.device, "device")
        self.model = network
        print(self.model)
        print("Learning rate:", args.learning_rate)

        self.loss_fn = nn.MSELoss()

        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=args.learning_rate,
                                         momentum=args.momentum,
                                         nesterov=args.nesterov,
                                         weight_decay=args.weight_decay)

        self.has_learned_something = False

        self.losses = []

    def eval_batch(self, ss):
        return np.array([self.eval(s) for s in ss])

    def eval(self, s):
        if not self.has_learned_something or s is None:
            return 0
        return float(self.predict(s).max())

    def best(self, s):
        if not self.has_learned_something or s is None:
            return 0
        return int(self.predict(s).argmax())

    def predict(self, s):
        if not self.has_learned_something or s is None:
            return 0
        return self.model(torch.tensor(s).to(self.device))

    def single_update(self, s, value):
        return self.batch_update(np.array([s]), np.array([value]))

    def batch_update(self, ss, values):

        ss = torch.tensor(ss).to(self.device)
        values = torch.tensor(values, dtype=torch.float, device=self.device).reshape(len(ss), 1)

        self.optimizer.zero_grad()
        pred = self.model(ss)

        loss = self.loss_fn(pred, values)
        loss.backward()
        self.optimizer.step()

        self.losses.append(loss.item())
        self.has_learned_something = True

    def to_onnx(self):
        x = torch.randn(1, self.nfeatures, device=self.device)
        torch.onnx.export(self.model,  # model being run
                          x,  # model input (or a tuple for multiple inputs)
                          "tmp.onnx",  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=10,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['X'],  # the model's input names
                          output_names=['output'],  # the model's output names
                          dynamic_axes={'X': {0: 'batch_size'},  # variable length axes
                                        'output': {0: 'batch_size'}})
        return onnx.load("tmp.onnx"), InferenceSession("tmp.onnx")

    def nfeatures(self):
        return self.nfeatures

    # should be called only at the end of each episode
    def current_loss(self):
        avg_loss = np.mean(self.losses)
        self.losses = []
        return avg_loss


class NeuralNetwork(nn.Module):
    def __init__(self, nfeatures, nnsize):
        super(NeuralNetwork, self).__init__()
        nnsize = list(nnsize) + [1]
        layers = [nn.Linear(nfeatures, nnsize[0])]
        for i in range(len(nnsize)-1):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(nnsize[i], nnsize[i+1]))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def reuse_onnx_model(self, onnx_path):
        raise NotImplementedError