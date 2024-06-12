import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import cross_val_score
import math

np.random.seed(42)


class PSOGWO():
    def __init__(self, X_train, X_test, y_train, y_test, dim, SearchAgents_no, T, ub, lb,
                 init_w, k):

        self.X_train = X_train  # 训练集
        self.X_test = X_test  # 测试集
        self.y_train = y_train  # 训练结果
        self.y_test = y_test  # 测试结果
        self.D = dim  # 变量个数
        self.P = SearchAgents_no  # 种群大小
        self.G = T  # 迭代次数
        self.ub = np.array(ub)  # 上限
        self.lb = np.array(lb)  # 下限
        self.init_w = init_w  # 自身权重因子
        self.k = k
        self.v_max = self.k * (self.ub - self.lb)

        self.gbest_X = np.zeros([self.D])
        self.gbest_F = np.inf
        self.F_alpha = np.inf
        self.F_beta = np.inf
        self.F_delta = np.inf
        self.X_alpha = np.zeros([self.D])
        self.X_beta = np.zeros([self.D])
        self.X_delta = np.zeros([self.D])

    def opt(self):
        self.X = np.random.uniform(
            low=self.lb, high=self.ub, size=[self.P, self.D])
        self.V = np.random.uniform(
            low=self.lb, high=self.ub, size=[self.P, self.D])
        for i in range(self.P):
            for j in range(self.D):
                if i == 0:
                    self.X[i][j] = np.random.uniform(
                        low=self.lb[j], high=self.ub[j])
                    self.V[i][j] = np.random.uniform(
                        low=self.lb[j], high=self.ub[j])
                else:
                    self.X[i][j] = (4 * self.X[i - 1][j] + 0.5 - 0.8 / (2 * math.pi) * math.sin(
                        2 * math.pi * self.X[i - 1][j])) % 1 * (self.ub[j] - self.lb[j])
                    self.V[i][j] = (4 * self.X[i - 1][j] + 0.5 - 0.8 / (2 * math.pi) * math.sin(
                        2 * math.pi * self.X[i - 1][j])) % 1 * (self.ub[j] - self.lb[j])

        self.ng_best_X = self.gbest_X.copy()
        self.ng_best_F = 200
        iterations = []
        accuracy = []
        for g in range(self.G):

            F = np.empty((len(self.X)))
            for i in range(len(self.X)):
                F[i] = self.fitness(self.X[i])

            if np.min(F) < self.gbest_F:
                idx = F.argmin()
                self.gbest_X = self.X[idx].copy()
                self.gbest_F = F.min()

            if self.gbest_F < self.ng_best_F:
                self.ng_best_X = self.gbest_X
                self.ng_best_F = self.gbest_F



            for i in range(self.P):
                if F[i] < self.F_alpha:
                    self.F_alpha = F[i].copy()
                    self.X_alpha = self.X[i].copy()
                elif F[i] < self.F_beta:
                    self.F_beta = F[i].copy()
                    self.X_beta = self.X[i].copy()
                elif F[i] < self.F_delta:
                    self.F_delta = F[i].copy()
                    self.X_delta = self.X[i].copy()

            a = 2 - math.cos(math.pi / 2 * (g / self.G))
            self.w = self.init_w + np.random.uniform() / 3

            r1 = np.random.uniform(size=[self.P, self.D])
            r2 = np.random.uniform(size=[self.P, self.D])
            A = 2 * a * r1 - a
            C1 = 2 * r2
            D_Alpha = np.abs(C1 * self.X_alpha - self.w * self.X)
            X1 = self.X_alpha - A * D_Alpha

            r1 = np.random.uniform(size=[self.P, self.D])
            r2 = np.random.uniform(size=[self.P, self.D])
            A = 2 * a * r1 - a
            C2 = 2 * r2
            D_Beta = np.abs(C2 * self.X_beta - self.w * self.X)
            X2 = self.X_beta - A * D_Beta

            r1 = np.random.uniform(size=[self.P, self.D])
            r2 = np.random.uniform(size=[self.P, self.D])
            A = 2 * a * r1 - a
            C3 = 2 * r2
            D_Delta = np.abs(C3 * self.X_delta - self.w * self.X)
            X3 = self.X_delta - A * D_Delta

            r2 = np.random.uniform(size=[self.P, self.D])
            r3 = np.random.uniform(size=[self.P, self.D])
            r4 = np.random.uniform(size=[self.P, self.D])

            self.V = self.w * (self.V + C1 * r2 * (X1 - self.X)
                               + C2 * r3 * (X2 - self.X)
                               + C3 * r4 * (X3 - self.X))
            self.V = np.clip(self.V, -self.v_max, self.v_max)
            self.X = (X1 + X2 + X3) / 3
            self.X = self.X + self.V
            self.X = np.clip(self.X, self.lb, self.ub)
            iterations.append(g)
            accuracy.append((100 - self.ng_best_F) / 100)
        best_C = self.ng_best_X[0]
        best_gamma = self.ng_best_X[1]
        return best_C, best_gamma, iterations, accuracy

    # 适应度函数
    def fitness(self, position):
        # fitness
        # pop中的适应度函数
        rbf_regressor = svm.SVR(kernel='rbf', C=position[0], gamma=position[1]).fit(
            self.X_train, self.y_train)  # svm
        cv_accuracies = cross_val_score(rbf_regressor, self.X_test, self.y_test, cv=3,
                                        scoring='neg_mean_squared_error')
        accuracies = cv_accuracies.mean()
        fitness_value = (1 - accuracies) * 100
        return fitness_value

    def return_result(self):

        return self.result
