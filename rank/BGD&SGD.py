import numpy as np
import random

def batchGradientDescent(x, y, theta, alpha, m, maxInteration):
    x_train = x.transpose()
    for i in range(0 , maxInteration):
        hypothesis = np.dot(x, theta)

        loss = hypothesis - y

        gradient = np.dot(x_train, loss) / m

        theta = theta - alpha * gradient
    return theta

def main():
    trainData = np.array([[1,4,2],[2,5,3],[5,1,6],[4,2,8]])
    trainLabel = np.array([19,26,19,20])
    print(trainData)
    print(trainLabel)
    m,n = np.shape(trainData)
    theta = np.ones(n)
    print(m,n)
    x_train = trainData.transpose()
    print(x_train)

if __name__ == '__main__':
    main()

