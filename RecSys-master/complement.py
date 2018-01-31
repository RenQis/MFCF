import numpy as np

def culculate_average(R):
    average_ui = []
    average_dj = []
    count_ui = []
    count_dj = []
    for i in range(R.shape[0]):
        count = 0
        sum_ui = 0
        for j in range(R.shape[1]):
            if R[i][j] != 0:
                count += 1
                sum_ui += R[i][j]
        if count == 0:
            average_ui.append(0)
        else:
            average_ui.append(sum_ui / float(count))
        count_ui.append(count)
    for j in range(R.shape[1]):
        count = 0
        sum_dj = 0
        for i in range(R.shape[0]):
            if R[i][j] != 0:
                count += 1
                sum_dj += R[i][j]
        if count == 0:
            average_dj.append(0)
        else:
            average_dj.append(sum_dj / float(count))
        count_dj.append(count)
    return average_ui, average_dj, count_ui, count_dj

def complement(R):
    average_ui, average_dj, count_ui, count_dj = culculate_average(R)
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if R[i][j] == 0:
                count_ij = count_ui[i] + count_dj[j]
                if count_ij == 0:
                    R[i][j] = 0
                else:
                    ratio_ui = count_ui[i] / float(count_ij)
                    ratio_dj = count_dj[j] / float(count_ij)
                    R[i][j] = ratio_ui * average_ui[i] + ratio_dj * average_dj[j]
    return R

def create_auxiliary_matrix(R, alpha):
    M = [[1 for col in range(R.shape[1])] for row in range(R.shape[0])]
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if R[i][j] == 0:
                M[i][j] = alpha
    M = np.array(M)
    return M
