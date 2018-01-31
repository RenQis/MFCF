import nmf1
import nmf2
from complement import *
from preprocessing import create_datasets
from metric import evaluate
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    dataset1, dataset2, dataset3, dataset4, dataset5 = create_datasets()
    #print(dataset1)
    print ('************datasets were created!!***************')

    R_train1 = np.array(dataset1[0])
    R_test1 = np.array(dataset1[1])
    R_train2 = np.array(dataset2[0])
    R_test2 = np.array(dataset2[1])
    R_train3 = np.array(dataset3[0])
    R_test3 = np.array(dataset3[1])
    R_train4 = np.array(dataset4[0])
    R_test4 = np.array(dataset4[1])
    R_train5 = np.array(dataset5[0])
    R_test5 = np.array(dataset5[1])
    #print(R_train1,R_test1)
    K = 19 # number of latent features
    # initialize P and Q with random values
    P = np.random.rand(R_train1.shape[0], K)
    Q = np.random.rand(R_train1.shape[1], K)
    print(R_test1.shape[0],R_test1.shape[1])

    print ('***************************************')
    print ('NMF1 started!!')
    print ('***************************************')

    P1, Q1 = nmf1.matrix_factorization(R_train1, P, Q, K)
    R_pred1 = np.dot(P1, Q1.T)
    #print(R_pred1)
    precision1, recall1, fvalue1, rmse1 = evaluate(R_test1, R_pred1)
    P2, Q2 = nmf1.matrix_factorization(R_train2, P, Q, K)
    R_pred2 = np.dot(P2, Q2.T)
    precision2, recall2, fvalue2, rmse2 = evaluate(R_test2, R_pred2)
    P3, Q3 = nmf1.matrix_factorization(R_train3, P, Q, K)
    R_pred3 = np.dot(P3, Q3.T)
    precision3, recall3, fvalue3, rmse3 = evaluate(R_test3, R_pred3)
    P4, Q4 = nmf1.matrix_factorization(R_train4, P, Q, K)
    R_pred4 = np.dot(P4, Q4.T)
    precision4, recall4, fvalue4, rmse4 = evaluate(R_test4, R_pred4)
    P5, Q5 = nmf1.matrix_factorization(R_train5, P, Q, K)
    R_pred5 = np.dot(P5, Q5.T)
    precision5, recall5, fvalue5, rmse5 = evaluate(R_test5, R_pred5)

    # 5 cross validation
    print ('**************precision***************')
    print ((precision1 + precision2 + precision3 + precision4 + precision5) / 5.0)
    print ('**************recall***************')
    print ((recall1 + recall2 + recall3 + recall4 + recall5) / 5.0)
    print ('**************fvalue***************')
    print ((fvalue1 + fvalue2 + fvalue3 + fvalue4 + fvalue5) / 5.0)
    print ('**************rmse***************')
    print ((rmse1 + rmse2 + rmse3 + rmse4 + rmse5) / 5.0)

    print ('***************************************')
    print ('NMF1 finished!!')
    print ('***************************************')

    """
    M1_nmf2 = create_auxiliary_matrix(R_train1, 0.1)
    M2_nmf2 = create_auxiliary_matrix(R_train2, 0.1)
    M3_nmf2 = create_auxiliary_matrix(R_train3, 0.1)
    M4_nmf2 = create_auxiliary_matrix(R_train4, 0.1)
    M5_nmf2 = create_auxiliary_matrix(R_train5, 0.1)
    M1_nmf3 = create_auxiliary_matrix(R_train1, 0.7)
    M2_nmf3 = create_auxiliary_matrix(R_train2, 0.7)
    M3_nmf3 = create_auxiliary_matrix(R_train3, 0.7)
    M4_nmf3 = create_auxiliary_matrix(R_train4, 0.7)
    M5_nmf3 = create_auxiliary_matrix(R_train5, 0.7)

    print ('*****auxiliary_matrix was created!!*****')

    print ('***************************************')
    print ('NMF2 with 0 started!!')
    print ('***************************************')

    P1, Q1 = nmf2.matrix_factorization(R_train1, M1_nmf2, P, Q, K, 0.05)
    R_pred1 = np.dot(P1, Q1.T)
    precision1, recall1, fvalue1, rmse1 = evaluate(R_test1, R_pred1)
    P2, Q2 = nmf2.matrix_factorization(R_train2, M2_nmf2, P, Q, K, 0.05)
    R_pred2 = np.dot(P2, Q2.T)
    precision2, recall2, fvalue2, rmse2 = evaluate(R_test2, R_pred2)
    P3, Q3 = nmf2.matrix_factorization(R_train3, M2_nmf2, P, Q, K, 0.05)
    R_pred3 = np.dot(P3, Q3.T)
    precision3, recall3, fvalue3, rmse3 = evaluate(R_test3, R_pred3)
    P4, Q4 = nmf2.matrix_factorization(R_train4, M4_nmf2, P, Q, K, 0.05)
    R_pred4 = np.dot(P4, Q4.T)
    precision4, recall4, fvalue4, rmse4 = evaluate(R_test4, R_pred4)
    P5, Q5 = nmf2.matrix_factorization(R_train5, M5_nmf2, P, Q, K, 0.05)
    R_pred5 = np.dot(P5, Q5.T)
    precision5, recall5, fvalue5, rmse5 = evaluate(R_test5, R_pred5)

    # 5 cross validation
    print ('**************precision***************')
    print ((precision1 + precision2 + precision3 + precision4 + precision5) / 5.0)
    print ('**************recall***************')
    print ((recall1 + recall2 + recall3 + recall4 + recall5) / 5.0)
    print ('**************fvalue***************')
    print ((fvalue1 + fvalue2 + fvalue3 + fvalue4 + fvalue5) / 5.0)
    print ('**************rmse***************')
    print ((rmse1 + rmse2 + rmse3 + rmse4 + rmse5) / 5.0)

    print ('***************************************')
    print ('NMF2 with 0 finished!!')
    print ('***************************************')

    R_train1 = complement(R_train1)
    R_train2 = complement(R_train2)
    R_train3 = complement(R_train3)
    R_train4 = complement(R_train4)
    R_train5 = complement(R_train5)

    print ('************complemented!!***************')

    print ('***************************************')
    print ('NMF2 with expected rating started!!')
    print ('***************************************')

    P1, Q1 = nmf2.matrix_factorization(R_train1, M1_nmf3, P, Q, K, 0.01)
    R_pred1 = np.dot(P1, Q1.T)
    precision1, recall1, fvalue1, rmse1 = evaluate(R_test1, R_pred1)
    P2, Q2 = nmf2.matrix_factorization(R_train2, M2_nmf3, P, Q, K, 0.01)
    R_pred2 = np.dot(P2, Q2.T)
    precision2, recall2, fvalue2, rmse2 = evaluate(R_test2, R_pred2)
    P3, Q3 = nmf2.matrix_factorization(R_train3, M2_nmf3, P, Q, K, 0.01)
    R_pred3 = np.dot(P3, Q3.T)
    precision3, recall3, fvalue3, rmse3 = evaluate(R_test3, R_pred3)
    P4, Q4 = nmf2.matrix_factorization(R_train4, M4_nmf3, P, Q, K, 0.01)
    R_pred4 = np.dot(P4, Q4.T)
    precision4, recall4, fvalue4, rmse4 = evaluate(R_test4, R_pred4)
    P5, Q5 = nmf2.matrix_factorization(R_train5, M5_nmf3, P, Q, K, 0.01)
    R_pred5 = np.dot(P5, Q5.T)
    precision5, recall5, fvalue5, rmse5 = evaluate(R_test5, R_pred5)

    # 5 cross validation
    print ('**************precision***************')
    print ((precision1 + precision2 + precision3 + precision4 + precision5) / 5.0)
    print ('**************recall***************')
    print ((recall1 + recall2 + recall3 + recall4 + recall5) / 5.0)
    print ('**************fvalue***************')
    print ((fvalue1 + fvalue2 + fvalue3 + fvalue4 + fvalue5) / 5.0)
    print ('**************rmse***************')
    print ((rmse1 + rmse2 + rmse3 + rmse4 + rmse5) / 5.0)

    print ('***************************************')
    print ('NMF2 with average rating finished!!')
    print ('***************************************')
    """
