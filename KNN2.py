import numpy as np
from numpy.linalg import norm
import heapq
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
import time

tr_img = np.load("tr_img.npy")
tr_lab = np.load("tr_lab.npy")
t10k_img = np.load("t10k_img.npy")
t10k_lab = np.load("t10k_lab.npy")

class KNN:
    global tr_img, tr_lab, t10k_lab, t10k_img
    def __init__(self, k, default_dist):
        self.k = k
        self.learn_label_wrong = []
        self.distance = []
        if default_dist == "man":
            self.calc_dist = self.manhattan
        elif default_dist == "euc":
            self.calc_dist == "euc"

    def manhattan(self):
        # self.distance = [[norm(tr_img[j] - t10k_img[i]) for j in range(60000)] for i in range(10000)] #too slow
        self.distance = manhattan_distances(t10k_img, tr_img) # 700s

    def euclid(self):
        self.distance = euclidean_distances(t10k_img, tr_img)

    def predict(self):
        count = 0
        indi_wrong_lab_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        indi_truth_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.calc_dist()
        minimum = [heapq.nsmallest(self.k, range(len(i)), i.take) for i in self.distance]
        min_label = [[tr_lab[j] for j in i] for i in minimum]
        predict_label = [self.mode(i) for i in min_label]
        for i in range(len(t10k_lab)):
            for j in range(10):
                if t10k_lab[i] == j:
                    indi_truth_count[j] += 1
                    if predict_label[i] != t10k_lab[i]:
                        count += 1
                        indi_wrong_lab_count[j] += 1
        for i in range(10):
            print(i, " Success Rate:", 1 - indi_wrong_lab_count[i] / indi_truth_count[i])
        print("Total Success Rate:", 1 - count / 10000)

    def mode(self, a):
        single = []
        count = []
        for i in a:
            if i not in single:
                single.append(i)
                count.append(1)
            else:
                count[single.index(i)] += 1
        max_count = 0
        mode = 0
        for i in count:
            if i > max_count:
                max_count = i
                mode = single[count.index(i)]

        return mode

knn = KNN(3, "man")
start = time.time()
knn.predict()
print(time.time() - start)

