import time
import numpy as np
import matplotlib.pyplot as plt

"""
NMF with complemented matrix
proposed NMF: missing ratings is complemented by expected ratings
Devooght's NMF: missing ratings is complemented by 0
"""

def matrix_factorization(R, M, P, Q, K, gamma, lambda_=0.1, steps=30):
    start = time.time()
    Q = Q.T
    #x = []
    #y = []
    for step in range(steps):
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                eij = (R[i][j] - np.dot(P[i,:], Q[:,j])) * M[i][j]
                for k in range(K):
                    gradient_p = -2 * eij * Q[k][j] + 2 * lambda_ * P[i][k]
                    gradient_q = -2 * eij * P[i][k] + 2 * lambda_ * Q[k][j]
                    if P[i][k] - gamma * gradient_p >= 0:
                        P[i][k] = P[i][k] - gamma * gradient_p
                    if Q[k][j] - gamma * gradient_q >= 0:
                        Q[k][j] = Q[k][j] - gamma * gradient_q
        eR = np.dot(P, Q)
        e = 0
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                e += pow(R[i][j] - eR[i][j], 2) * M[i][j]
                for k in range(K):
                    e += lambda_ * (pow(P[i][k], 2) + pow(Q[k][j], 2))
        e = e / 10000.0
        #x.append(step)
        #y.append(e)
        if e < 0.1:
            break
    #plt.plot(x, y)
    #plt.show()
    elapsed_time = time.time() - start
    print (("elapsed_time:{0}".format(elapsed_time)) + "[sec]")
    return P, Q.T
