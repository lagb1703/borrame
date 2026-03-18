from datasets import load_dataset, load_dataset_builder
import numpy as np
from typing import Tuple
import os
import math
from tqdm import tqdm

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
    split: Tuple[int, int], 
    shape=(224,224), 
    label: str = "train",
    tqdmDisable: bool = True,
    classNumber = 1000):
    shard, index = split
    wbw = batchSize//workerNumber
    newShape = shape[0]*shape[1]
    x = np.zeros((wbw, newShape))
    y = np.zeros((wbw, classNumber))
    dataset = ds.shard(num_shards=shard, index=index)
    size = size // workerNumber
    try:
        folder = os.path.join('.', f'data-{label}-{wbw}')
        img, label_img = labels
        xFolder = os.path.join(folder, 'x')
        yFolder = os.path.join(folder, 'y')
        if not os.path.exists(folder):
            os.mkdir(folder)
            os.mkdir(xFolder)
            os.mkdir(yFolder)
        for i in tqdm(range(math.floor(size/batchSize)), disable=tqdmDisable, desc="batch"):
            path_x = os.path.join(xFolder, f"batch-{shard}-{i*(index+1)}.npy")
            path_y = os.path.join(yFolder, f"batch-{shard}-{i*(index+1)}.npy")
            if os.path.exists(path_x) and os.path.exists(path_y):
                x = np.load(path_x)
                y = np.load(path_y)
                dataset = dataset.skip(wbw)
                yield (x, y)
                continue
            for j, element in enumerate(dataset.take(wbw)):
                image = element[img].resize(shape)
                image_label = element[label_img]
                image = np.array(image, dtype=np.float32)
                if image.ndim == 3 and image.shape[2] >= 3:
                    image = (
                        image[:, :, 0] * 0.299 +
                        image[:, :, 1] * 0.587 +
                        image[:, :, 2] * 0.114
                    ) / 255.0
                elif image.ndim == 2:
                    image = image / 255.0
                else:
                    image = np.mean(image, axis=2) / 255.0 if image.ndim == 3 else image / 255.0
                x[j] = image.reshape((newShape))
                y[j] = one_hot_encode(image_label, classNumber)
            np.save(path_x, x)
            np.save(path_y, y)
            dataset = dataset.skip(wbw)
            yield (x, y)
            x = np.zeros((wbw, newShape))
            y = np.zeros((wbw, classNumber))
    except Exception as e:
        print(e)
        yield (x, y)