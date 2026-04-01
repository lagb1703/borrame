from datasets import load_dataset, load_dataset_builder
import numpy as np
from typing import Tuple, List, Callable, Any
import os
import math
from tqdm import tqdm
from socket import socket

def downloadDataset(dataset: str, splitName: str):
    token = os.getenv("token")
    ds = load_dataset(dataset, split=splitName, token=token, streaming=True)
    builder = load_dataset_builder(dataset, token=token,)
    total = builder.info.splits[splitName].num_examples
    return (ds, total)

def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]

def getBatch(
    ds, 
    batchSize:int, 
    labels: Tuple[str, str], 
    size: int, 
    workerNumber: int,
    split: int, 
    shape=(224,224), 
    label: str = "train",
    tqdmDisable: bool = True,
    classNumber = 1000):
    wbw = batchSize // workerNumber 
    newShape = shape[0] * shape[1]
    totalSteps = math.ceil(size / batchSize)
    img_key, label_key = labels
    folder = os.path.join('.', f'data-{label}-shard{split}')
    xFolder = os.path.join(folder, 'x')
    yFolder = os.path.join(folder, 'y')
    if not os.path.exists(xFolder): os.makedirs(xFolder)
    if not os.path.exists(yFolder): os.makedirs(yFolder)
    for i in tqdm(range(totalSteps), disable=tqdmDisable, desc=f"Shard {split} Batches"):
        path_x = os.path.join(xFolder, f"batch-{i}.npy")
        path_y = os.path.join(yFolder, f"batch-{i}.npy")
        if os.path.exists(path_x) and os.path.exists(path_y):
            yield (np.load(path_x), np.load(path_y))
            continue
        offset = (i * batchSize) + (split * wbw)
        current_shard_ds = ds.skip(offset).take(wbw)
        x = np.zeros((wbw, newShape), dtype=np.float32)
        y = np.zeros((wbw, classNumber), dtype=np.float32)
        for j, element in enumerate(current_shard_ds):
            image = element[img_key].resize(shape)
            image = np.array(image, dtype=np.float32)
            if image.ndim == 3:
                image = (image @ [0.299, 0.587, 0.114]) / 255.0
            else:
                image = image / 255.0
            x[j] = image.flatten()
            y[j] = one_hot_encode(element[label_key], classNumber)
        np.save(path_x, x)
        np.save(path_y, y)
        yield (x, y)
        
def sigmoidea(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def devSigmoidea(x: np.ndarray) -> np.ndarray:
    s = sigmoidea(x)
    return s * (1.0 - s)

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)

def devRelu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, 1.0, 0.0)

def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def devSoftmax(x: np.ndarray) -> np.ndarray:
    s = softmax(x)
    s_vec = s.reshape(-1)
    jacobian_matrix = np.diag(s_vec) - np.outer(s_vec, s_vec)
    return jacobian_matrix

def mse(predicted: np.ndarray, actually: np.ndarray) -> np.ndarray:
    return np.mean((predicted - actually) ** 2)

def devMse(predicted: np.ndarray, actually: np.ndarray) -> np.ndarray:
    n = predicted.size
    return np.where(n > 0, (2.0 / n) * (predicted - actually), np.zeros_like(predicted))

def lostEntropy(predicted: np.ndarray, actually: np.ndarray) -> np.ndarray:
    eps = 1e-7
    p = np.clip(predicted, eps, 1.0 - eps)
    return -np.sum(actually * np.log(p))

def devLostEntropy(predicted: np.ndarray, actually: np.ndarray) -> np.ndarray:
    eps = 1e-7
    p = np.clip(predicted, eps, 1.0 - eps)
    return -(actually / p)

class Layer:
    def __init__(
            self,
            neurons: int,
            activation: Callable[[np.ndarray], np.ndarray],
            derivada: Callable[[np.ndarray], np.ndarray],
            name: str):
        self.neurons: int = neurons
        self.activacion = activation
        self.derivada = derivada
        self.name: str = name
        
    def export(self)->str:
        return f"{self.name}-{self.neurons}-{self.activacion.__name__}-{self.derivada.__name__}"
    
    def __str__(self)->str:
        return f"{self.name}-{self.neurons}-{self.activacion.__name__}-{self.derivada.__name__}"
    
    @staticmethod
    def load(layer: str)->'Layer':
        name, neurons_str, activationName, derivadaName = layer.split("-")
        neurons = int(neurons_str)
        activation = None
        derivada = None
        match(activationName):
            case "sigmoidea":
                activation = sigmoidea
            case "relu":
                activation = relu
            case "softmax":
                activation = softmax
        
        match(derivadaName):
            case "devSigmoidea":
                derivada = devSigmoidea
            case "devRelu":
                derivada = devRelu
            case "devSoftmax":
                derivada = devSoftmax
        return Layer(neurons, activation, derivada, name)

class Model:
    def __init__(
            self,
            sequential: Any,
            w: List[np.ndarray],
            b: List[np.ndarray]):
        self.sequential = sequential
        self.set_parameters(w, b)
        
    def set_parameters(self, w: List[np.ndarray], b: List[np.ndarray]):
        self.w = [np.array(weights, dtype=np.float32) for weights in w]
        self.b = [np.array(bias, dtype=np.float32) for bias in b]
        
    def getParams(self):
        return (self.w, self.b)

    def fordward(self, x: np.ndarray) -> np.ndarray:
      neu = [None] * len(self.sequential)
      z = [None] * len(self.sequential)
      neu[0] = x
      for i in range(1, len(self.sequential)):
        neu[i] = self.sequential[i].activacion(np.dot(neu[i - 1], self.w[i - 1]) + self.b[i])
      return neu[-1]

class Sequential:
    def __init__(self, *layers: Layer):
        self.layers = layers

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, i):
        return self.layers[i]
    
    def export(self):
        export = ""
        nLayer = len(self.layers)
        for index, i in enumerate(self.layers):
            if index < nLayer - 1:
                export += f"{i.export()}\n"
            else:
                export += f"{i.export()}"
        return export
    
    @staticmethod
    def load(layers: str)->'Sequential':
        internal = []
        for i in layers.split("\n"):
            internal.append(Layer.load(i))
        return Sequential(*internal)
    
def __evaluate(neu, x_sample: np.ndarray, z, sequential, w, b):
    neu[0] = x_sample
    for i in range(1, len(sequential)):
        z[i] = np.dot(neu[i - 1], w[i - 1]) + b[i]
        neu[i] = sequential[i].activacion(z[i])

def evaluate(w, b, x_test, y_test, sequential, error):
    correct_predictions = 0
    errors = []
    for x_b, y_b in zip(x_test, y_test):
        neu_v = [None] * len(sequential)
        z_v = [None] * len(sequential)
        __evaluate(neu_v, x_b, z_v, sequential, w, b)
        errors.append(error(neu_v[-1], y_b))
        if np.argmax(neu_v[-1]) == np.argmax(y_b):
            correct_predictions += 1
    return (correct_predictions, errors)

def __forward(neu, x_sample: np.ndarray, z, sequential, w, b):
    neu[0] = x_sample
    for i in range(1, len(sequential)):
        z[i] = np.dot(neu[i - 1], w[i - 1]) + b[i]
        neu[i] = sequential[i].activacion(z[i])

def __backward(dEdz, z, sequential, w) -> np.ndarray:
    for i in range(len(sequential) - 2, 0, -1):
        dEdz[i] = (dEdz[i+1] @ w[i].T) * sequential[i].derivada(z[i])
        
def batch(x_b, y_b, w, b, w_grad_batch, b_grad_batch, sequential, devError):
    neu = [None] * len(sequential)
    z = [None] * len(sequential)
    __forward(neu, x_b, z, sequential, w, b)
    dEdz = [None] * len(sequential)
    de = devError(neu[-1], y_b)
    if de.shape == (1,):
      dEdz[-1] = de * sequential[-1].derivada(z[-1])
    else:
      dEdz[-1] = de @ sequential[-1].derivada(z[-1])
    __backward(dEdz, z, sequential, w)
    for i in range(len(sequential) - 1):
        w_grad_sample = np.outer(neu[i], dEdz[i+1])
        w_grad_batch[i] += w_grad_sample
        b_grad_batch[i+1] += dEdz[i+1]

def recvall(sock: socket, n: int) -> bytearray:
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            raise ConnectionError("Conexión cerrada antes de recibir todos los datos esperados")
        data.extend(packet)
    return data