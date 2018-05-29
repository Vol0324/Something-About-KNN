import numpy as np
from numpy.linalg import norm
import struct
import matplotlib.pyplot as plt

path_tr_img = "MNIST/train-images.idx3-ubyte"
binary_tr_img = open(path_tr_img, "rb").read()
path_tr_lab = "MNIST/train-labels.idx1-ubyte"
binary_tr_lab = open(path_tr_lab, "rb").read()
path_t10k_img = "MNIST/t10k-images.idx3-ubyte"
binary_t10k_img = open(path_t10k_img, "rb").read()
path_t10k_lab = "MNIST/t10k-labels.idx1-ubyte"
binary_t10k_lab = open(path_t10k_lab, "rb").read()

tr_img = []
tr_lab = []
t10k_img = []
t10k_lab = []

for i in range(60000):
    img = np.array(struct.unpack_from(">784B", binary_tr_img, 784 * i + 16))
    # img = img.reshape(28, 28)
    tr_img.append(img)
    lab = struct.unpack_from(">B", binary_tr_lab, 8 + i)
    tr_lab.append(lab[0])

tr_lab = np.array(tr_lab)
np.save("tr_lab.npy", tr_lab)
tr_img = np.array(tr_img)
np.save("tr_img.npy", tr_img)


for i in range(10000):
    img = np.array(struct.unpack_from(">784B", binary_t10k_img, 784 * i + 16))
    # img = img.reshape(28, 28)
    t10k_img.append(img)
    lab = struct.unpack_from(">B", binary_t10k_lab, 8 + i)
    t10k_lab.append(lab[0])

tr_lab = np.array(t10k_lab)
np.save("t10k_lab.npy", tr_lab)
tr_img = np.array(t10k_img)
np.save("t10k_img.npy", tr_img)
