#coding=utf-8
#输入：样本集D = {X1,X2,...,Xm)
#      聚类蔟数k
import numpy as np

def loadData():
    pass

def distance(xi, xj):
    """
    计算两个向量间的距离
    :param xi: 向量1
    :param xj: 向量2
    :return: 距离度量，这里用的平方误差
    """
    return np.sum((np.asarray(xi) - np.asarray(xj))**2)

def meanPoint(C):
    """
    :param C: 聚类蔟集合
    :return: 这个蔟内的均值向量
    """
    return np.mean(C, axis = 0, keepdims=False)

def randCenter(D, k):
    """
    产生随机的中心点用于聚类, 随机点需要在数据集的边界内
    :param D: 数据集D
    :param k: 聚类数k
    :return: 中心点集合centerGroup
    """
    n = np.shape(D)[1]
    centerGroup = np.zeros((k,n))
    for j in range(n):
        minJ = min(D[:, j])
        rangeJ = float(max(D[:, j]) - minJ)
        centerGroup[:,j] = minJ + rangeJ * np.random.rand(k, 1)
    return centerGroup


def kmeans(k, D, centerGroup, clusterAssignment, dist = distance):
    """
    实现kmeans算法, 这段代码在所有机器上运行
    :param k: 聚类数量k
    :param D: 输入数据集D
    :param dist: 距离度量，默认为平方误差，可自行定义
    :param centerGroup: 当前均值向量
    :param clusterAssignment: 存储每个点的蔟分配结果及其离center的距离, 每次执行需要传入上次执行结果
    :return: 聚类结果
    """

    num = len(D)

    # 标记均值向量是否有更新。有则继续聚类，否则退出聚类
    clusterChanged = False
    for i in range(num):
        minDist = np.inf
        minIndex = -1
        for j in range(k):
            distJI = dist(D[i], D[j])
            if distJI < minDist:
                minDist = distJI
                minIndex = j
        #分配结果发生改变说明均值向量会发生变化，需要继续聚类
        if clusterAssignment[i, 0] != minIndex:
            clusterChanged = True
        clusterAssignment[i,:] = minIndex, minDist
    #更新均值向量
    for center in range(k):
        pointsIncenter = D[clusterAssignment[:,0] == center]
        centerGroup[center] = meanPoint(pointsIncenter)
    return clusterChanged, centerGroup, clusterAssignment #对于主机器以外的机器将结果传回主机器


def kmeansTogether(centerGroupGroup):
    """
    处理从各个机器传送过来的结果，更新均值向量，此段代码只在主机器上运行
    :param centerGroupGroup: 各个机器返回的局部均值向量集合，对其取均值得到全局均值向量
    :return: centerGroup 均值向量集合，索引值为所代表类别
    """
    return meanPoint(centerGroupGroup)

def main():
    """
    主机器执行代码
    :return: centerGroup-质心点结果，clusterAssignment-每个点对应的质心分配结果
    """
    """
    将数据分配到各个(z个)分布式节点
    D2 = loadData()
    ...
    Dz = loadData()
    其中，Di（i=1,2...z）为m/z个n维连续变量的数据集，默认第一台机器为主机器
    """

    Di = loadData()

    z = 10
    k = 5
    n = np.shape(Di)[1]

    centerGroup = randCenter(Di, k = 5)

    """
    产生随机质心，并传输到每台机器上，即所有机器运算时使用的质心是一致的
    """

    clusterChanged = True
    clusterAssignmenti = np.zeros([len(Di), 2])

    while clusterChanged:

        clusterChangedi, centerGroupi, newclusterAssignmenti = kmeans(k=5, D=Di, centerGroup=centerGroup, clusterAssignment = clusterAssignmenti)
        """
        接受其他机器的结果， 并将其结果合并：
        """
        clusterChanged1 = None
        clusterChanged2 = None
        clusterChangedz = None
        clusterChanged = clusterChanged1 & clusterChanged2 & ... & clusterChangedz
        centerGroupGroup = np.zeros((z, k, n))
        for i in range(z):
            centerGroupGroup[i] = centerGroupi
        centerGroup = kmeansTogether(centerGroupGroup)
        """
        将新质心传输到每台机器上
        """
    clusterAssignment = newclusterAssignment0
    for i in range(1, z):
        np.concatenate([clusterAssignment, newclusterAssignmenti])

    return centerGroup, clusterAssignment

if __name__ == '__main__':
    main()

