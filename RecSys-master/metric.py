import math
import numpy as np
import copy

def evaluate(R_test, R_pred):
    """
    for all (i, j) pairs
    tp is the number of R_test[i][j] >= 4 and R_pred[i][j] >= 4
    fp is the number of R_test[i][j] < 4 and R_pred[i][j] >= 4
    tn is the number of R_test[i][j] < 4 and R_pred[i][j] < 4
    fn is the number of R_test[i][j] >= 4 and R_pred[i][j] < 4
    """
    #_max = 0
    a = []
    #for i in range(R_pred.shape[0]):
     #   for j in range(R_pred.shape[1]):
      #      if _max < R_pred[i][j]:
       #         _max = R_pred[i][j]
    """
    pred = R_pred.copy()
    for x in range(10):
        raw,column = pred.shape
        _position = np.argmax(pred)
        m, n = divmod(_position, column)
        a.append(pred[m][n])
        pred[m][n] = 0
    print(a)
    """
    pre = 0.0
    rec = 0.0
    fva = 0.0
    tp = fp = tn = fn = 0
    count = 0
    squared_error = 0
    pred = R_pred.copy()
    x = 0
    k = 40
    for (row1,row2) in zip(pred,R_test):
        for i in row2:
            if i == 1:
                fp += 1
        for i in range(k):
            #raw,column = pred.shape
            y = np.argmax(row1)
            #m, n = divmod(_position, column)
            #a.append(pred[m][n])
            #pred[m][n] = 0
            #print(R_test[x][y])
            row1[y] = 0
            if R_test[x][y] == 1:
                tp += 1 
            else:
                fn += 1
        print(fp,tp,fn)
        if fp != 0:
            pre += tp / float(k)
            rec += tp / float(fp)
            if (pre + rec) != 0:
                fva += 2 * pre * rec /float(pre + rec)
            tn = len(R_test)
        else:
            tn = len(R_test) - 1
        x = x + 1
        fp = tp = fn = 0
    p = pre / tn
    r = rec / tn
    f = fva / tn
    
    for i in range(R_test.shape[0]):
        for j in range(R_test.shape[1]):
            if R_test[i][j] != 0:
                count += 1
                squared_error += pow(R_test[i][j] - R_pred[i][j], 2)
                """
                if R_test[i][j] >= 1:
                    if R_pred[i][j] >= 1:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if R_pred[i][j] >= 1:
                        fp += 1
                    else:
                        tn += 1
                """
    #p = precision(tp, fp)
    #r = recall(tp, fn)
    #f = fvalue(p, r)
    e = rmse(squared_error, count)
    return p, r, f, e
    
    """
    inv_lst = np.unique(R_test[:, 0])
    pred = {}
    print(inv_lst)
    for inv in inv_lst:
        if pred.get(inv, None) is None:
            pred[inv] = np.argsort(R_pred(inv))[-k:]  # numpy.argsort索引排序
            #print(self.predict(inv))
            #print(pred[inv])

    intersection_cnt = {}
    for i in range(R_test.shape[0]):
        if R_test[i, 1] in pred[R_test[i, 0]]:
            intersection_cnt[R_test[i, 0]] = intersection_cnt.get(R_test[i, 0], 0) + 1
    invPairs_cnt = np.bincount(np.array(R_test[:, 0], dtype='int32'))

    precision_acc = 0.0
    recall_acc = 0.0
    for inv in inv_lst:
        precision_acc += intersection_cnt.get(inv, 0) / float(k)
        recall_acc += intersection_cnt.get(inv, 0) / float(invPairs_cnt[int(inv)])
        #print(intersection_cnt.get(inv, 0))
        #print(float(invPairs_cnt[int(inv)]))

    return precision_acc/ len(inv_lst) , recall_acc / len(inv_lst),len(inv_lst)
    """

    
def precision(tp, fp):
    if tp + fp == 0:
        return 0
    else:
        return tp / float(tp + fp)

def recall(tp, fn):
    if tp + fn == 0:
        return 0
    else:
        return tp / float(tp + fn)

def fvalue(p, r):
    if p + r == 0:
        return 0
    else:
        return 2 * p * r / float(p + r)

def rmse(se, count):
    return math.sqrt(se / float(count))
