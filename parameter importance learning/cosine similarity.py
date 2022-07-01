from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])
res = cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)) 