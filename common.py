

# def getBatch(ds, batchSize:int, labels: Tuple[str, str], shape=(224,224), classNumber = 10000):
#     img, label = labels
#     newShape = shape[0]*shape[1]
#     x = np.zeros((batchSize, newShape))
#     y = np.zeros((batchSize, classNumber))
#     try:
#         for i in range()
#         for index, element in enumerate(ds):
#             image = element[img].resize(shape)
#             image_label = element[label]
#             i = index%batchSize
#             image = np.array(image, dtype=np.float32)
#             image = (
#                 image[:, :, 0] * 0.299 +
#                 image[:, :, 1] * 0.587 +
#                 image[:, :, 2] * 0.114
#             ) / 255.0
#             x[i] += image.reshape((newShape))
#             y[i] += one_hot_encode(image_label, classNumber)
#             if i == 0:
#                 yield (x, y)
#                 x = np.zeros((batchSize, newShape))
#                 y = np.zeros((batchSize, classNumber))
#     except Exception as e:
#         yield (x, y)