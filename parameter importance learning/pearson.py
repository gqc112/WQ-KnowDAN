import math

import numpy as np


def pearson(v1, v2):
    n = len(v1)
    #simple sums
    sum1 = sum(float(v1[i]) for i in range(n))
    sum2 = sum(float(v2[i]) for i in range(n))
    #sum up the squares
    sum1_pow = sum([pow(v, 2.0) for v in v1])
    sum2_pow = sum([pow(v, 2.0) for v in v2])
    #sum up the products
    p_sum = sum([v1[i] * v2[i] for i in range(n)])
    #分子num，分母denominator
    num = p_sum - (sum1*sum2/n)
    den = math.sqrt((sum1_pow-pow(sum1, 2)/n)*(sum2_pow-pow(sum2, 2)/n))
    if den == 0:
        return 0.0
    return num/den

vector1 = [2,7,18,88,157, 90,177,570]
vector2 = [3,5,15,90,180, 88,160,580]

print(pearson(vector2,vector1))
