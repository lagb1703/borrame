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
    shape: Tuple[int, int]=(224,224), 
    label: str = "train",
    tqdmDisable: bool = True,
    classNumber: int = 1000):
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
        
def sigmoidea(x: np.ndarray[float, np.dtype[Any]]) -> np.ndarray[float, np.dtype[Any]]:
    return 1.0 / (1.0 + np.exp(-x))

def devSigmoidea(x: np.ndarray[float, np.dtype[Any]]) -> np.ndarray[float, np.dtype[Any]]:
    s = sigmoidea(x)
    return s * (1.0 - s)

def relu(x: np.ndarray[float, np.dtype[Any]]) -> np.ndarray[float, np.dtype[Any]]:
    return np.maximum(x, 0.0)

def devRelu(x: np.ndarray[float, np.dtype[Any]]) -> np.ndarray[float, np.dtype[Any]]:
    return np.where(x > 0, 1.0, 0.0)

def softmax(x: np.ndarray[float, np.dtype[Any]]) -> np.ndarray[float, np.dtype[Any]]:
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def devSoftmax(x: np.ndarray[float, np.dtype[Any]]) -> np.ndarray[float, np.dtype[Any]]:
    s = softmax(x)
    s_vec = s.reshape(-1)
    jacobian_matrix = np.diag(s_vec) - np.outer(s_vec, s_vec)
    return jacobian_matrix

def mse(predicted: np.ndarray[float, np.dtype[Any]], actually: np.ndarray[float, np.dtype[Any]]) -> float:
    return float(np.mean((predicted - actually) ** 2))

def devMse(predicted: np.ndarray[float, np.dtype[Any]], actually: np.ndarray[float, np.dtype[Any]]) -> np.ndarray[float, np.dtype[Any]]:
    n = predicted.size
    return np.where(n > 0, (2.0 / n) * (predicted - actually), np.zeros_like(predicted))

def lostEntropy(predicted: np.ndarray[float, np.dtype[Any]], actually: np.ndarray[float, np.dtype[Any]]) -> float:
    eps = 1e-7
    p = np.clip(predicted, eps, 1.0 - eps)
    return -float(np.sum(actually * np.log(p)))

def devLostEntropy(predicted: np.ndarray[float, np.dtype[Any]], actually: np.ndarray[float, np.dtype[Any]]) -> np.ndarray[float, np.dtype[Any]]:
    eps = 1e-7
    p = np.clip(predicted, eps, 1.0 - eps)
    return -(actually / p)

class Layer:
    def __init__(
            self,
            neurons: int,
            activation: Callable[[np.ndarray[float, np.dtype[Any]]], np.ndarray[float, np.dtype[Any]]],
            derivada: Callable[[np.ndarray[float, np.dtype[Any]]], np.ndarray[float, np.dtype[Any]]],
            name: str):
        self.neurons: int = neurons
        self.activacion: Callable[[np.ndarray[float, np.dtype[Any]]], np.ndarray[float, np.dtype[Any]]] = activation
        self.derivada: Callable[[np.ndarray[float, np.dtype[Any]]], np.ndarray[float, np.dtype[Any]]] = derivada
        self.name: str = name
        
    def export(self)->str:
        return f"{self.name}-{self.neurons}-{self.activacion.__name__}-{self.derivada.__name__}"
    
    def __str__(self)->str:
        return f"{self.name}-{self.neurons}-{self.activacion.__name__}-{self.derivada.__name__}"
    
    @staticmethod
    def load(layer: str)->'Layer':
        name, neurons_str, activationName, derivadaName = layer.split("-")
        neurons = int(neurons_str)
        activation: Callable[[np.ndarray[float, np.dtype[Any]]], np.ndarray[float, np.dtype[Any]]] | None = None
        derivada: Callable[[np.ndarray[float, np.dtype[Any]]], np.ndarray[float, np.dtype[Any]]] = None
        match(activationName):
            case "sigmoidea":
                activation = sigmoidea
            case "relu":
                activation = relu
            case "softmax":
                activation = softmax
            case _:
                print("none")
        match(derivadaName):
            case "devSigmoidea":
                derivada = devSigmoidea
            case "devRelu":
                derivada = devRelu
            case "devSoftmax":
                derivada = devSoftmax
            case _:
                print("none")
        if activation is None or derivada is None:
            raise
        return Layer(neurons, activation, derivada, name)


class Sequential:
    def __init__(self, *layers: Layer):
        self.layers = layers

    def __len__(self)->int:
        return len(self.layers)

    def __getitem__(self, i: int)->Layer:
        return self.layers[i]
    
    @staticmethod
    def load(layers: str)->'Sequential':
        internal:List[Layer] = []
        for i in layers.split("\n"):
            internal.append(Layer.load(i))
        return Sequential(*internal)
    
    def export(self):
        export = ""
        nLayer = len(self.layers)
        for index, i in enumerate(self.layers):
            if index < nLayer - 1:
                export += f"{i.export()}\n"
            else:
                export += f"{i.export()}"
        return export
    
    def __forward(
        self, 
        neu: List[None | np.ndarray[float, np.dtype[Any]]], 
        x_sample: np.ndarray[float, np.dtype[Any]], 
        z: List[None | np.ndarray[float, np.dtype[Any]]], 
        w: np.ndarray[float, np.dtype[Any]], 
        b: np.ndarray[float, np.dtype[Any]])->None:
        neu[0] = x_sample
        for i in range(1, len(self)):
            z[i] = np.dot(neu[i - 1], w[i - 1]) + b[i] # type: ignore
            neu[i] = self[i].activacion(z[i]) # type: ignore

    def __backward(
        self, 
        dEdz: List[None | np.ndarray[float, np.dtype[Any]]], 
        z: List[None | np.ndarray[float, np.dtype[Any]]], 
        w: List[np.ndarray[float, np.dtype[Any]]])->None:
        for i in range(len(self) - 2, 0, -1):
            dEdz[i] = (dEdz[i+1] @ w[i].T) * self[i].derivada(z[i]) # type: ignore

    def batch(
        self, 
        x_b: np.ndarray[float, np.dtype[Any]], 
        y_b: np.ndarray[float, np.dtype[Any]], 
        w: List[np.ndarray[float, np.dtype[Any]]], 
        b: List[np.ndarray[float, np.dtype[Any]]], 
        w_grad_batch: List[np.ndarray[float, np.dtype[Any]]], 
        b_grad_batch: List[np.ndarray[float, np.dtype[Any]]], 
        devError: Callable[[np.ndarray[float, np.dtype[Any]], np.ndarray[float, np.dtype[Any]]], np.ndarray[float, np.dtype[Any]]]):
        neu: List[None | np.ndarray[float, np.dtype[Any]]] = [None] * len(self)
        z: List[None | np.ndarray[float, np.dtype[Any]]] = [None] * len(self)
        self.__forward(neu, x_b, z, w, b)
        dEdz: List[None | np.ndarray[float, np.dtype[Any]]] = [None] * len(self)
        de = devError(neu[-1], y_b) # type: ignore
        if de.shape == (1,):
            dEdz[-1] = de * self[-1].derivada(z[-1]) # type: ignore
        else:
            dEdz[-1] = de @ self[-1].derivada(z[-1]) # type: ignore
        self.__backward(dEdz, z, w)
        for i in range(len(self) - 1):
            w_grad_sample: np.ndarray[float, np.dtype[Any]] = np.outer(neu[i], dEdz[i+1]) # type: ignore
            w_grad_batch[i] += w_grad_sample
            b_grad_batch[i+1] += dEdz[i+1]

class Model:
    def __init__(
            self,
            sequential: Sequential,
            w: List[np.ndarray[float, np.dtype[Any]]],
            b: List[np.ndarray[float, np.dtype[Any]]]):
        self.sequential: Sequential = sequential
        self.set_parameters(w, b)
        
    def set_parameters(self, w: List[np.ndarray[float, np.dtype[Any]]], b: List[np.ndarray[float, np.dtype[Any]]]):
        self.w = [np.array(weights, dtype=np.float32) for weights in w]
        self.b = [np.array(bias, dtype=np.float32) for bias in b]
        
    def getParams(self):
        return (self.w, self.b)

    def fordward(self, x: np.ndarray[float, np.dtype[Any]]) -> np.ndarray[float, np.dtype[Any]]:
      neu: List[None | np.ndarray[float, np.dtype[Any]]] = [None] * len(self.sequential)
      neu[0] = x
      for i in range(1, len(self.sequential)):
        neu[i] = self.sequential[i].activacion(np.dot(neu[i - 1], self.w[i - 1]) + self.b[i]) # type: ignore
      return neu[-1] # type: ignore
    
    
def __evaluate(
    neu: List[np.ndarray[float, np.dtype[Any]] | None], 
    x_sample: np.ndarray[float, np.dtype[Any]], 
    z: np.ndarray[float, np.dtype[Any]], 
    sequential: Sequential, 
    w: List[np.ndarray[float, np.dtype[Any]]], 
    b: List[np.ndarray[float, np.dtype[Any]]]):
    neu[0] = x_sample
    for i in range(1, len(sequential)):
        z[i] = np.dot(neu[i - 1], w[i - 1]) + b[i] # type: ignore
        neu[i] = sequential[i].activacion(z[i])

def evaluate(
    w: List[np.ndarray[float, np.dtype[Any]]], 
    b: List[np.ndarray[float, np.dtype[Any]]], 
    x_test: np.ndarray[float, np.dtype[Any]], 
    y_test: np.ndarray[float, np.dtype[Any]], 
    sequential: Sequential, 
    error: Callable[[np.ndarray[float, np.dtype[Any]], np.ndarray[float, np.dtype[Any]]], float]):
    correct_predictions = 0
    errors:List[float] = []
    for x_b, y_b in zip(x_test, y_test):
        neu_v = [None] * len(sequential)
        z_v = [None] * len(sequential)
        __evaluate(neu_v, x_b, z_v, sequential, w, b) # type: ignore
        errors.append(error(neu_v[-1], y_b)) # type: ignore
        if np.argmax(neu_v[-1]) == np.argmax(y_b): # type: ignore
            correct_predictions += 1
    return (correct_predictions, errors)

def recvall(sock: socket, n: int) -> bytearray:
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            raise ConnectionError("Conexión cerrada antes de recibir todos los datos esperados")
        data.extend(packet)
    return data