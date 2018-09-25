# 感知机学习模型
# 输入为数据集(x1,y1),(x2,y2)...(xn,yn),其中xi为R维特征向量，yi为类别{+1,-1}
# 输出为能将数据集正确分开的超平面w*x+b=0的参数w,b
# 学习算法：随机梯度下降法，stochastic gradient descent
# 损失函数L(w,b)=-Σyi*(w*xi+b),极小化结果为0.(当数据集线性可分时)

import numpy as np


def perceptron_learning(X,Y,yita):
    # η为梯度下降速度，取值为(0,1]，learning rate
    if len(X) != len(Y):
        print("The X and Y must have same dimension.")
        return -1
    w = np.array([0]*len(X[0]))
    b = 0
    finish_signal = False
    count = 0
    max_iter = 1000 # 最大迭代次数,由(R/γ)^2确定
    while finish_signal is False:
        for i in range(0,len(X)):
            if (w.dot(X[i])+b)*Y[i]<=0:
                w = w+yita*(np.array(Y[i])*np.array(X[i]))
                b = b+yita*Y[i]
                print(w,b)
                i  = 0
                count += 1
                if count == max_iter:
                    print("It has reached the maximum iteration.")
                    return -2
            if i == len(X)-1:
                finish_signal = True
    return w,b

# 感知机模型的对偶形式
# 由于w=w+η*xi*yi,可视为第i个数据对w的一次修正
# 求解α = (a1,a2...,an)，为每个数据修正的次数
# 优势在于可提前计算Gram矩阵，减少矩阵乘法运算过程
def perceptron_pairs(X,Y,yita):
    alpha = np.array([0]*len(X))
    b = 0
    count = 0
    finish_signal = False

    while finish_signal is False:
        for i in range(0,len(X)):
            w = np.array([0]*len(X[0]))
            b2 = 0
            for j in range(len(X)): # 计算此条件下的w和b
                w = w+np.array(alpha[j])*np.array(Y[j])*np.array(X[j])
                b2 = b2+alpha[j]*Y[j]
            if (w.dot(X[i])+b)*Y[i] <= 0:
                # print(alpha,b)
                alpha[i] = alpha[i] + yita
                b = b+yita*Y[i]
                i = 0
                count += 1
                if count == 100:
                    print("Maximum iteration.")
                    return -2
            if i == len(X)-1:
                # print(w,b)
                finish_signal = True
    # print(alpha,b,count)
    return alpha,b


# perceptron_learning([[3,3,5],[4,3,8],[1,1,9]],[1,1,-1],1)

# perceptron_pairs([[3,3],[4,3],[1,1]],[1,1,-1],1)
