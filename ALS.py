import random
import csv
import numpy as np
import math


def init():
    data = []
    rating_count = []
    with open("data.csv") as f:
        for line in f.readlines():
            r = line.split(",")
            r = [float(i) for i in r]
            rating_count.append(int(r[0]))
            data.append(r[1:])
    data = data[:1000]
    rating_count = rating_count[:1000]

    train_data = [[99 for i in range(len(data[0]))] for j in range(len(data))]
    validate_data = [[99 for i in range(len(data[0]))] for j in range(len(data))]
    test_data = [[99 for i in range(len(data[0]))] for j in range(len(data))]
    full_train_data = [[99 for i in range(len(data[0]))] for j in range(len(data))]

    for i in range(len(data)):
        train_part = int(rating_count[i] * 0.6)
        validate_part = int(rating_count[i] * 0.8)
        test_part = rating_count[i]

        data1 = []
        for j in range(len(data[0])):
            if data[i][j] != 99:
                data1.append(j)

        random.shuffle(data1)
        train = data1[0:train_part]
        validate = data1[train_part:validate_part]
        test = data1[validate_part:test_part]

        for j in train:
            train_data[i][j] = data[i][j]
            full_train_data[i][j] = data[i][j]
        for j in validate:
            validate_data[i][j] = data[i][j]
            full_train_data[i][j] = data[i][j]
        for j in test:
            test_data[i][j] = data[i][j]

    writer = csv.writer(open("train.csv", 'w', newline=''))
    writer.writerows(train_data)
    writer = csv.writer(open("validate.csv", 'w', newline=''))
    writer.writerows(validate_data)
    writer = csv.writer(open("test.csv", 'w', newline=''))
    writer.writerows(test_data)
    writer = csv.writer(open("full_train.csv", 'w', newline=''))
    writer.writerows(full_train_data)

    return train_data, validate_data, test_data, full_train_data


def update_v(U_T, l_v, I_K, X, C_M, K):
    #print("In Update V")
    #print("Found U_T", U_T)
    V = np.zeros((len(X[0]), K))
    for m in range(len(X[0])):
        c_m = C_M[m]
        sum = np.zeros((K, K))
        sum2 = np.zeros((K, 1))
        for i in c_m:
            u_t = np.asarray(U_T[i]).reshape(1, K)
            u = np.asarray(u_t.transpose())
            val = np.matmul(u, u_t)
            sum += val
            sum2 += np.asarray(X[i][m]*u).reshape((K, 1))
        sum += l_v*I_K
        sum = np.linalg.inv(sum)

        v = np.matmul(sum, sum2)
        V[m] = v.transpose()
    #print("Returning V", V)
    return V


def update_u(V, l_u, I_K, X, C_N, K):
    #print("In Update U")
    #print("Found V", V)
    U_T = np.zeros((len(X), K))
    for n in range(len(X)):
        c_n = C_N[n]
        sum = np.zeros((K, K))
        sum2 = np.zeros((K, 1))
        for i in c_n:
            v_t = np.asarray(V[i]).reshape(1, K)
            v = np.asarray(v_t.transpose())
            val = np.matmul(v, v_t)
            sum += val
            sum2 += np.asarray(X[n][i]*v).reshape((K, 1))
        sum += l_u*I_K
        sum = np.linalg.inv(sum)

        u = np.matmul(sum, sum2)
        u_t = u.transpose()
        U_T[n] = u_t
    #print("Returning U_T", U_T)
    return U_T


def calc_rmse(X, X_t):
    rmse = 0
    cnt = 0
    for i in range(len(X)):
        for j in range(len(X[0])):
            if X[i][j] != 99:
                rmse += ((X[i][j] - X_t[i][j]) * (X[i][j] - X_t[i][j]))
                cnt += 1
    rmse = rmse/cnt
    rmse = math.sqrt(rmse)
    return rmse


def parameter_estimation():
    print("Parameter Estimation")
    validate_data = []

    with open("validate.csv") as f:
        for line in f.readlines():
            r = line.split(",")
            r = [float(i) for i in r]
            validate_data.append(r)

    validate_data = np.asarray(validate_data)

    train_d = []
    with open("train.csv") as f:
        for line in f.readlines():
            r = line.split(",")
            r = [float(i) for i in r]
            train_d.append(r)
    train_d = np.asarray(train_d)

    X = train_d
    M = len(X[0])
    N = len(X)
    X = np.array(X)
    rmse = 1000000
    l_u_train = 0
    l_v_train = 0
    K_train = 0
    C_M = [[] for _ in range(M)]
    C_N = [[] for _ in range(N)]

    for i in range(N):
        for j in range(M):
            if X[i][j] != 99:
                C_M[j].append(i)
                C_N[i].append(j)
    for l_u in [0.01, 0.1, 1, 10]:
        for l_v in [0.01, 0.1, 1, 10]:
            for K in [5, 10, 20, 40]:
                print("Run : ", l_u, l_v, K)
                I_K = np.identity(K)
                U_T = [[np.random.uniform(-10, 10) for _ in range(K)] for _ in range(N)]
                U_T = np.asarray(U_T)
                V = np.zeros((M, K))
                #print("Initial")
                #print(U_T)
                #print(V)
                for x in range(5):
                    #print("Running")
                    V = update_v(U_T, l_v, I_K, X, C_M, K)
                    #print("Updated V")
                    #print(V)
                    U_T = update_u(V, l_u, I_K, X, C_N, K)
                    #print("Updated U_T")
                    #print(U_T)
                U_T = np.asarray(U_T)
                V = np.asarray(V)
                #print("End")
                #print(U_T)
                #print(V)
                X_t = np.matmul(U_T, V.transpose())
                rmse_cur = calc_rmse(validate_data, X_t)
                print(rmse_cur)
                if rmse_cur < rmse:
                    rmse = rmse_cur
                    l_u_train = l_u
                    l_v_train = l_v
                    K_train = K

    print(rmse)
    print(l_u_train, l_v_train, K_train)
    return l_u_train, l_v_train, K_train


def train(K_train, l_u_train, l_v_train):
    print("Train")
    data = []

    with open("full_train.csv") as f:
        for line in f.readlines():
            r = line.split(",")
            r = [float(i) for i in r]
            data.append(r)

    data = np.asarray(data)

    X = data
    M = len(X[0])
    N = len(X)
    X = np.array(X)
    K = K_train
    l_u = l_u_train
    l_v = l_v_train
    I_K = np.identity(K)
    C_M = [[] for _ in range(M)]
    C_N = [[] for _ in range(N)]

    for i in range(N):
        for j in range(M):
            if data[i][j] != 99:
                C_M[j].append(i)
                C_N[i].append(j)
    U_T = [[np.random.uniform(-10, 10) for _ in range(K)] for _ in range(N)]
    U_T = np.asarray(U_T)
    V = np.zeros((M, K))
    for x in range(5):
        V = update_v(U_T, l_v, I_K, X, C_M, K)
        U_T = update_u(V, l_u, I_K, X, C_N, K)
    U_T = np.asarray(U_T)
    V = np.asarray(V)
    file = open("U.txt", 'w')
    for i in range(len(U_T)):
        for j in range(len(U_T[0])):
            file.write(str(U_T[i][j]) + " ")
        file.write("\n")
    file.close()

    file = open("V.txt", 'w')
    for i in range(len(V)):
        for j in range(len(V[0])):
            file.write(str(V[i][j]) + " ")
        file.write("\n")
    file.close()


def test():
    print("Test")
    U_T = []
    with open('U.txt') as f:
        lines = f.readlines()
    f.close()
    for i in lines:
        x = i.replace('\n', '').split(' ')
        x = [float(i) for i in x[:len(x) - 1]]
        U_T.append(x)
    V = []
    with open('V.txt') as f:
        lines = f.readlines()
    f.close()
    for i in lines:
        x = i.replace('\n', '').split(' ')
        x = [float(i) for i in x[:len(x) - 1]]
        V.append(x)
    V = np.asarray(V)
    data = []

    with open("test.csv") as f:
        for line in f.readlines():
            r = line.split(",")
            r = [float(i) for i in r]
            data.append(r)

    data = np.asarray(data)

    X_t = np.matmul(U_T, V.transpose())
    rmse = calc_rmse(data, X_t)
    print(rmse)


train_data, validate_data, test_data, full_train_data = init()
l_u_train, l_v_train, K_train = parameter_estimation()
train(K_train, l_u_train, l_v_train)
test()



