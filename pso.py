import csv
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import explained_variance_score
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import explained_variance_score
from sklearn import metrics
from sklearn.metrics import mean_absolute_error  # 平方绝对误差
import random
from sklearn import svm


class PSO:
    def __init__(self, X_train, X_test, y_train, y_test, SearchAgents_no, T, dim, lb, ub):
        """
        particle swarm optimization
        parameter: a list type, like [NGEN, pop_size, var_num_min, var_num_max]
        """
        # 初始化
        self.X_train = X_train  # 训练集
        self.X_test = X_test  # 测试集
        self.y_train = y_train  # 训练结果
        self.y_test = y_test  # 测试结果
        self.NGEN = T  # 迭代的代数
        self.pop_size = SearchAgents_no  # 种群大小
        self.var_num = dim  # 变量个数
        self.bound = []  # 变量的约束范围
        self.lb = lb  # 下限
        self.ub = ub  # 上限

        self.pop_x = np.zeros((self.pop_size, self.var_num))  # 所有粒子的位置
        self.pop_v = np.zeros((self.pop_size, self.var_num))  # 所有粒子的速度
        self.p_best = np.zeros((self.pop_size, self.var_num))  # 每个粒子最优的位置
        self.g_best = np.zeros((1, self.var_num))  # 全局最优的位置

        # 初始化第0代初始全局最优解
        temp = 200
        for i in range(self.pop_size):
            for j in range(self.var_num):
                self.pop_x[i][j] = random.uniform(self.lb[j], self.ub[j])
                self.pop_v[i][j] = random.uniform(0, 1)
            self.p_best[i] = self.pop_x[i]  # 储存最优的个体
            # fitness
            rbf_regressor = svm.SVR(kernel='rbf', C=self.p_best[i][0], gamma=self.p_best[i][1]).fit(self.X_train,
                                                                                                    self.y_train)  # svm
            cv_accuracies = cross_val_score(rbf_regressor, self.X_test, self.y_test, cv=3,
                                            scoring='neg_mean_squared_error')
            accuracies = cv_accuracies.mean()
            fitness_value = (1 - accuracies) * 100
            fit = fitness_value
            if fit < temp:
                self.g_best = self.p_best[i]
                temp = fit


    def update_operator(self, pop_size):
        """
        更新算子：更新下一时刻的位置和速度
        """
        c1 = 1.2  # 学习因子，一般为2
        c2 = 1.2
        w = 0.5  # 自身权重因子
        for i in range(pop_size):
            # 更新速度
            self.pop_v[i] = w * self.pop_v[i] + c1 * random.uniform(0, 1) * (
                    self.p_best[i] - self.pop_x[i]) + c2 * random.uniform(0, 1) * (self.g_best - self.pop_x[i])
            # 更新位置
            self.pop_x[i] = self.pop_x[i] + self.pop_v[i]
            # 越界保护
            for j in range(self.var_num):
                if self.pop_x[i][j] < self.lb[j]:
                    self.pop_x[i][j] = self.lb[j]
                if self.pop_x[i][j] > self.ub[j]:
                    self.pop_x[i][j] = self.ub[j]
            # 更新p_best和g_best
            # fitness
            # pop中的适应度函数
            rbf_regressor_p = svm.SVR(kernel='rbf', C=self.pop_x[i][0], gamma=self.pop_x[i][1]).fit(
                self.X_train, self.y_train)  # svm
            cv_accuracies = cross_val_score(rbf_regressor_p, self.X_test, self.y_test, cv=3,
                                            scoring='neg_mean_squared_error')
            accuracies = cv_accuracies.mean()
            fitness_value_p = (1 - accuracies) * 100

            # 群体最优适应度函数
            rbf_regressor_b = svm.SVR(kernel='rbf', C=self.p_best[i][0], gamma=self.p_best[i][1]).fit(
                self.X_train,
                self.y_train)  # svm
            cv_accuracies = cross_val_score(rbf_regressor_b, self.X_test, self.y_test, cv=3,
                                            scoring='neg_mean_squared_error')
            accuracies = cv_accuracies.mean()
            fitness_value_b = (1 - accuracies) * 100


            # 个体最优适应度函数
            self.g_best = self.pop_x[i]
            rbf_regressor_g = svm.SVR(kernel='rbf', C=self.g_best[0], gamma=self.g_best[1]).fit(
                self.X_train,
                self.y_train)  # svm
            cv_accuracies = cross_val_score(rbf_regressor_g, self.X_test, self.y_test, cv=3,
                                            scoring='neg_mean_squared_error')
            accuracies = cv_accuracies.mean()
            fitness_value_g = (1 - accuracies) * 100
            if fitness_value_p < fitness_value_b:
                self.p_best[i] = self.pop_x[i]
            if fitness_value_p < fitness_value_g:
                self.g_best = self.pop_x[i]

    def main(self):
        iterations = []
        accuracy = []
        self.ng_best = self.g_best.copy()
        self.best_score = 200
        print(self.ng_best)
        for gen in range(self.NGEN):
            self.update_operator(self.pop_size)


            rbf_regressor_g = svm.SVR(kernel='rbf', C=self.g_best[0], gamma=self.g_best[1]).fit(
                self.X_train,
                self.y_train)  # svm
            cv_accuracies = cross_val_score(rbf_regressor_g, self.X_test, self.y_test, cv=3,
                                            scoring='neg_mean_squared_error')
            accuracies = cv_accuracies.mean()
            fitness_value_g = (1 - accuracies) * 100



            if fitness_value_g < self.best_score:
                self.best_score=fitness_value_g
                self.ng_best = self.g_best.copy()
            iterations.append(gen)
            accuracy.append((100 - self.best_score) / 100)
        best_C = self.ng_best[0]
        best_gamma = self.ng_best[1]
        return best_C, best_gamma,iterations, accuracy
