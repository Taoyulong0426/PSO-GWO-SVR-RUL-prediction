import numpy as np
from sklearn import svm
from sklearn.svm import SVR
import sklearn.model_selection
import numpy.random as rd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import warnings, pandas as pd, numpy as np, time, math, configparser, random


## 1. GWO optimization algorithm
class GWO:
    def __init__(self, X_train, X_test, y_train, y_test, SearchAgents_no, T, dim, lb, ub):
        self.X_train = X_train  # 训练集
        self.X_test = X_test  # 测试集
        self.y_train = y_train  # 训练结果
        self.y_test = y_test  # 测试结果
        self.SearchAgents_no = SearchAgents_no  # 狼群数量
        self.T = T  # 迭代次数
        self.dim = dim  # 寻优参数数量
        self.lb = lb  # 下限
        self.ub = ub  # 上限

    def sanitized_gwo(self):
        # 初始化o狼的位置
        Positions = np.zeros((self.SearchAgents_no, self.dim))

        for i in range(0, self.SearchAgents_no):
            for j in range(0, self.dim):
                Positions[i, j ] = np.random.rand() * (self.ub[j] - self.lb[j]) + self.lb[j]
        # 初始化ABD狼的位置
        Alpha_position = [0, 0]  # Initialize the position of Alpha Wolf
        Beta_position = [0, 0]
        Delta_position = [0, 0]

        Alpha_score = float("inf")  # Initialize the value of Alpha Wolf's objective function
        Beta_score = float("inf")
        Delta_score = float("inf")

        Convergence_curve = np.zeros((1, self.T))  # initialization fusion curve

        iterations = []
        accuracy = []

        # Main Loop
        t = 0
        while t < self.T:

            # Iterate over each wolf
            for i in range(0, (Positions.shape[0])):
                # If the search position exceeds the search space, you need to return to the search space
                for j in range(0, (Positions.shape[1])):
                    Flag4ub = Positions[i, j] > self.ub[j]
                    Flag4lb = Positions[i, j] < self.lb[j]
                    # If the wolf's position is between the maximum and minimum, the position does not need to be adjusted, if it exceeds the maximum, the maximum returns to the maximum value boundary

                    if Flag4ub:
                        Positions[i, j] = self.ub[j]
                    if Flag4lb:
                        Positions[i, j] = self.lb[j]
                '''SVM MODEL TRAINING - FOR CLASSIFICATION PROBLEM DATASET'''
                # rbf_svm = svm.SVC(kernel = 'rbf', C = Positions[i][0], gamma = Positions[i][1]).fit(X_train, y_train)  #svm
                # cv_accuracies = cross_val_score(rbf_svm,X_test,y_test,cv =3,scoring = 'accuracy')

                '''SVR MODEL TRAINING - FOR REGRESSION PROBLEM DATASET'''
                rbf_regressor = svm.SVR(kernel='rbf', C=Positions[i][0], gamma=Positions[i][1]).fit(self.X_train,
                                                                                                    self.y_train)  # svm
                cv_accuracies = cross_val_score(rbf_regressor, self.X_test, self.y_test, cv=3,
                                                scoring='neg_mean_squared_error')  # Taking negated value of MSE

                # To minimize the error rate
                accuracies = cv_accuracies.mean()
                fitness_value = (1 - accuracies) * 100
                if fitness_value < Alpha_score:  # If the objective function value is less than the objective function value of Alpha Wolf
                    Alpha_score = fitness_value  # Then update the target function value of Alpha Wolf to the optimal target function value
                    Alpha_position = Positions[
                        i]  # At the same time update the position of the Alpha wolf to the optimal position
                if fitness_value > Alpha_score and fitness_value < Beta_score:  # If the objective function value is between the objective function value of Alpha Wolf and Beta Wolf
                    Beta_score = fitness_value  # Then update the target function value of Beta Wolf to the optimal target function value
                    Beta_position = Positions[i]
                if fitness_value > Alpha_score and fitness_value > Beta_score and fitness_value < Delta_score:  # If the target function value is between the target function value of Beta Wolf and Delta Wolf
                    Delta_score = fitness_value  # Then update the target function value of Delta Wolf to the optimal target function value
                    Delta_position = Positions[i]

            a = 2 - t * (2 / self.T)

            # Iterate over each wolf
            for i in range(0, (Positions.shape[0])):
                # Traverse through each dimension
                for j in range(0, (Positions.shape[1])):
                    # Surround prey, location update
                    r1 = rd.random(1)  # Generate a random number between 0 ~ 1
                    r2 = rd.random(1)
                    A1 = 2 * a * r1 - a  # calculation factor A
                    C1 = 2 * r2  # calculation factor C
                    #C1 = 0.5 + (0.5 * math.exp(-j / 500)) + (
                    #        1.4 * (math.sin(j) / 30))  # Time varying Acceleration constant

                    # Alphawolf location update

                    D_alpha = abs(C1 * Alpha_position[j] - Positions[i, j])
                    X1 = Alpha_position[j] - A1 * D_alpha

                    r1 = rd.random(1)
                    r2 = rd.random(1)

                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    #C2 = 1 + (1.4 * (1 - math.exp(-j / 500))) + (
                    #        1.4 * (math.sin(j) / 30))  # Difference Mean based Perturbation time varying parameter

                    # Beta wolf location update
                    D_beta = abs(C2 * Beta_position[j] - Positions[i, j])
                    X2 = Beta_position[j] - A2 * D_beta
                    r1 = rd.random(1)
                    r2 = rd.random(1)

                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    # C3 = (1 / (1 + math.exp(-0.0001 * j / self.T))) + (
                    #     (0.5 - 2.5) * ((j / self.T) ** 2))  # sigmoid-based acceleration coefficient

                    # Delta Wolf Location Update
                    D_delta = abs(C3 * Delta_position[j] - Positions[i, j])
                    X3 = Delta_position[j] - A3 * D_delta

                    # Location update
                    Positions[i, j] = (X1 + X2 + X3) / 3

            t = t + 1
            iterations.append(t)
            accuracy.append((100 - Alpha_score) / 100)


        best_C = Alpha_position[0]
        best_gamma = Alpha_position[1]

        return best_C, best_gamma, iterations, accuracy
