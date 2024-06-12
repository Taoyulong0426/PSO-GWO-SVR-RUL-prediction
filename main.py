# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 21:35:27 2023

@author: pony
"""

# 导入所需的库
import pandas as pd  # 数据处理
import numpy as np  # 数值计算
import matplotlib.pyplot as plt  # 绘图
import seaborn as sns  # 数据可视化
from sklearn.ensemble import RandomForestRegressor  # 随机森林回归模型
import sklearn  # 机器学习库
from sklearn.metrics import mean_squared_error, r2_score  # 评估指标
from sklearn.linear_model import LogisticRegression
from gwo import GWO
import os  # 操作系统相关
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score  # 数据划分
from sklearn.preprocessing import StandardScaler  # 特征归一化
import random  # 随机数生成
import warnings  # 警告处理
import math
from pso import PSO
from psogwo import PSOGWO

np.random.seed(34)  # 设置随机种子
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['simhei']  # 添加中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False

# 列名定义
RUL_name = ['RUL']  # 寿命的名字
eigenvalue_names = ['Vol_cut_time', 'V_col_dec', 'vol_avr', 'cur_avr']  # 特征值名字
col_names = RUL_name + eigenvalue_names  # 第一列到最后一列的名字

# 读取打乱的数据集
dftrain_5 = pd.read_csv('./NASA_database/B0005.csv', sep=',', header=None, index_col=False, names=col_names)  # 读取训练集
dfvalid_5 = pd.read_csv('./NASA_database/B0005.csv', sep=',', header=None, index_col=False, names=col_names)  # 读取测试集

dftrain_6 = pd.read_csv('./NASA_database/B0006.csv', sep=',', header=None, index_col=False, names=col_names)  # 读取训练集
dfvalid_6 = pd.read_csv('./NASA_database/B0006.csv', sep=',', header=None, index_col=False, names=col_names)  # 读取测试集

dftrain_7 = pd.read_csv('./NASA_database/B0007.csv', sep=',', header=None, index_col=False, names=col_names)  # 读取训练集
dfvalid_7 = pd.read_csv('./NASA_database/B0007.csv', sep=',', header=None, index_col=False, names=col_names)  # 读取测试集

dftrain_18 = pd.read_csv('./NASA_database/B0018.csv', sep=',', header=None, index_col=False, names=col_names)  # 读取训练集
dfvalid_18 = pd.read_csv('./NASA_database/B0018.csv', sep=',', header=None, index_col=False, names=col_names)  # 读取测试集



# #读取按顺序的数据集
# dftrain_5 = pd.read_csv('./NASA_database/B0005_r.csv', sep=',', header=None, index_col=False, names=col_names)  # 读取训练集
# dfvalid_5 = pd.read_csv('./NASA_database/B0005.csv', sep=',', header=None, index_col=False, names=col_names)  # 读取测试集
#
# dftrain_6 = pd.read_csv('./NASA_database/B0006_r.csv', sep=',', header=None, index_col=False, names=col_names)  # 读取训练集
# dfvalid_6 = pd.read_csv('./NASA_database/B0006.csv', sep=',', header=None, index_col=False, names=col_names)  # 读取测试集
#
# dftrain_7 = pd.read_csv('./NASA_database/B0007_r.csv', sep=',', header=None, index_col=False, names=col_names)  # 读取训练集
# dfvalid_7 = pd.read_csv('./NASA_database/B0007.csv', sep=',', header=None, index_col=False, names=col_names)  # 读取测试集
#
# dftrain_18 = pd.read_csv('./NASA_database/B0018_r.csv', sep=',', header=None, index_col=False, names=col_names)  # 读取训练集
# dfvalid_18 = pd.read_csv('./NASA_database/B0018.csv', sep=',', header=None, index_col=False, names=col_names)  # 读取测试集


# 复制数据集
train_5 = dftrain_5.copy()
valid_5 = dfvalid_5.head(20).copy()

train_6 = dftrain_6.copy()
valid_6 = dfvalid_6.head(20).copy()

train_7 = dftrain_7.copy()
valid_7 = dfvalid_7.head(20).copy()

train_18 = dftrain_18.copy()
valid_18 = dfvalid_18.head(20).copy()

# 对数据归一化处理
from sklearn.model_selection import train_test_split  # 导入数据集划分函数
from sklearn.preprocessing import MinMaxScaler  # 导入特征缩放器

scaler = MinMaxScaler()  # 创建MinMaxScaler实例用于特征缩放

X_train_5, X_test_5, y_train_5, y_test_5 = train_test_split(train_5, train_5['RUL'], test_size=0.3,
                                                            random_state=42)  # 划分训练和测试数据集
X_train_6, X_test_6, y_train_6, y_test_6 = train_test_split(train_6, train_6['RUL'], test_size=0.3,
                                                            random_state=42)  # 划分训练和测试数据集
X_train_7, X_test_7, y_train_7, y_test_7 = train_test_split(train_7, train_7['RUL'], test_size=0.3,
                                                            random_state=42)  # 划分训练和测试数据集
X_train_18, X_test_18, y_train_18, y_test_18 = train_test_split(train_18, train_18['RUL'], test_size=0.3,
                                                                random_state=42)  # 划分训练和测试数据集
# 删除目标变量
X_train_5.drop(columns=['RUL'], inplace=True)
X_test_5.drop(columns=['RUL'], inplace=True)

X_train_6.drop(columns=['RUL'], inplace=True)
X_test_6.drop(columns=['RUL'], inplace=True)

X_train_7.drop(columns=['RUL'], inplace=True)
X_test_7.drop(columns=['RUL'], inplace=True)

X_train_18.drop(columns=['RUL'], inplace=True)
X_test_18.drop(columns=['RUL'], inplace=True)
# 缩放X_train和X_test特征
X_train_5_s = scaler.fit_transform(X_train_5)
X_test_5_s = scaler.fit_transform(X_test_5)

X_train_6_s = scaler.fit_transform(X_train_6)
X_test_6_s = scaler.fit_transform(X_test_6)

X_train_7_s = scaler.fit_transform(X_train_7)
X_test_7_s = scaler.fit_transform(X_test_7)

X_train_18_s = scaler.fit_transform(X_train_18)
X_test_18_s = scaler.fit_transform(X_test_18)

y_valid_5 = valid_5['RUL']
valid_5.drop(columns=['RUL'], inplace=True)
X_valid_5_s = scaler.fit_transform(valid_5)

y_valid_6 = valid_6['RUL']
valid_6.drop(columns=['RUL'], inplace=True)
X_valid_6_s = scaler.fit_transform(valid_6)

y_valid_7 = valid_7['RUL']
valid_7.drop(columns=['RUL'], inplace=True)
X_valid_7_s = scaler.fit_transform(valid_7)

y_valid_18 = valid_18['RUL']
valid_18.drop(columns=['RUL'], inplace=True)
X_valid_18_s = scaler.fit_transform(valid_18)


class Linear_Regression():
    def __init__(self, lr=0.01, iterations=150):
        self.lr = lr
        self.iterations = iterations

    def fit(self, X, Y):
        self.l, self.p = X.shape
        # 权重初始化
        self.W = np.zeros(self.p)
        self.b = 0
        self.X = X
        self.Y = Y
        # 梯度学习
        for i in range(self.iterations):
            self.weight_updater()
        return self

    def weight_updater(self):
        Y_pred = self.predict(self.X)
        # 计算梯度
        dW = - (2 * (self.X.T).dot(self.Y - Y_pred)) / self.l
        db = - 2 * np.sum(self.Y - Y_pred) / self.l
        # 更新权重
        self.b = self.b - self.lr * db
        self.W = self.W - self.lr * dW
        return self

    def predict(self, X):
        # Y_pred = X.W + b
        return X.dot(self.W) + self.b




# IGWO-SVR模型
# '''
# import xgboost  # 导入XGBoost库
#
# # 创建XGBoost回归器实例，设置超参数
# xgb = xgboost.XGBRegressor(n_estimators=110, learning_rate=0.02, gamma=0, subsample=0.8, colsample_bytree=0.5, max_depth=3)

#5号电池

import xgboost
# 电池5
rf_5 = xgboost.XGBRegressor(n_estimators=110, learning_rate=0.1, gamma=0.1, subsample=0.8, colsample_bytree=0.5, max_depth=3) # 创建随机森林回归器
rf_5.fit(X_train_5_s, y_train_5)  # 拟合随机森林模型
# 预测并评估
y_pred_rf_5 = rf_5.predict(X_valid_5_s)
mse_rf_5 = mean_squared_error(y_valid_5, y_pred_rf_5)
rmse_rf_5 = np.sqrt(mse_rf_5)
r2_rf_5 = r2_score(y_valid_5, y_pred_rf_5)
nrmse_rf_5 = rmse_rf_5 / (y_valid_5.max() - y_valid_5.min())
print("5号电池线性回归模型评价结果：")
print("RMSE =", rmse_rf_5)
print("MSE =", mse_rf_5)
print("Normalized RMSE=", nrmse_rf_5)
print("R Square =", r2_rf_5)

# 电池6
rf_6 = xgboost.XGBRegressor(n_estimators=110, learning_rate=0.1, gamma=0.2, subsample=0.4, colsample_bytree=0.5, max_depth=3)  # 创建随机森林回归器
rf_6.fit(X_train_6_s, y_train_6)  # 拟合随机森林模型
# 预测并评估
y_pred_rf_6 = rf_5.predict(X_valid_6_s)
mse_rf_6 = mean_squared_error(y_valid_6, y_pred_rf_6)
rmse_rf_6 = np.sqrt(mse_rf_6)
r2_rf_6 = r2_score(y_valid_6, y_pred_rf_6)
nrmse_rf_6 = rmse_rf_6 / (y_valid_6.max() - y_valid_6.min())
print("6号电池线性回归模型评价结果：")
print("RMSE =", rmse_rf_6)
print("MSE =", mse_rf_6)
print("Normalized RMSE=", nrmse_rf_6)
print("R Square =", r2_rf_6)

# 电池5
rf_7 = xgboost.XGBRegressor(n_estimators=110, learning_rate=0.1, gamma=0.2, subsample=0.4, colsample_bytree=0.5, max_depth=3)  # 创建随机森林回归器
rf_7.fit(X_train_7_s, y_train_7)  # 拟合随机森林模型
# 预测并评估
y_pred_rf_7 = rf_7.predict(X_valid_7_s)
mse_rf_7 = mean_squared_error(y_valid_7, y_pred_rf_7)
rmse_rf_7 = np.sqrt(mse_rf_7)
r2_rf_7 = r2_score(y_valid_7, y_pred_rf_7)
nrmse_rf_7 = rmse_rf_7 / (y_valid_7.max() - y_valid_7.min())
print("7号电池线性回归模型评价结果：")
print("RMSE =", rmse_rf_7)
print("MSE =", mse_rf_7)
print("Normalized RMSE=", nrmse_rf_7)
print("R Square =", r2_rf_7)

# 电池18
rf_18 = xgboost.XGBRegressor(n_estimators=110, learning_rate=0.1, gamma=0.2, subsample=0.4, colsample_bytree=0.5, max_depth=3) # 创建随机森林回归器
rf_18.fit(X_train_18_s, y_train_18)  # 拟合随机森林模型
# 预测并评估
y_pred_rf_18 = rf_18.predict(X_valid_18_s)
mse_rf_18 = mean_squared_error(y_valid_18, y_pred_rf_18)
rmse_rf_18 = np.sqrt(mse_rf_18)
r2_rf_18 = r2_score(y_valid_18, y_pred_rf_18)
nrmse_rf_18 = rmse_rf_18 / (y_valid_18.max() - y_valid_18.min())
print("5号电池线性回归模型评价结果：")
print("RMSE =", rmse_rf_18)
print("MSE =", mse_rf_18)
print("Normalized RMSE=", nrmse_rf_18)
print("R Square =", r2_rf_18)

'''
PSO-SVR模型
'''
from sklearn.svm import SVR

# 电池5
regressor_svr_5 = SVR(kernel='rbf', C=1, gamma=8)  # 创建SVR回归器
regressor_svr_5.fit(X_train_5_s, y_train_5)  # 拟合SVR模型
y_pred_svr_5 = regressor_svr_5.predict(X_valid_5_s)

mse_svr_5 = mean_squared_error(y_valid_5, y_pred_svr_5)
rmse_svr_5 = np.sqrt(mse_svr_5)
r2_svr_5 = r2_score(y_valid_5, y_pred_svr_5)
nrmse_svr_5 = rmse_svr_5 / (y_valid_5.max() - y_valid_5.min())
print("5号电池未优化的SVR模型评价结果：")
print("RMSE =", rmse_svr_5)
print("MSE =", mse_svr_5)
print("Normalized RMSE=", nrmse_svr_5)
print("R Square =", r2_svr_5)

# 电池6
regressor_svr_6 = SVR(kernel='rbf', C=1, gamma=8)  # 创建SVR回归器
regressor_svr_6.fit(X_train_6_s, y_train_6)  # 拟合SVR模型
y_pred_svr_6 = regressor_svr_6.predict(X_valid_6_s)

mse_svr_6 = mean_squared_error(y_valid_6, y_pred_svr_6)
rmse_svr_6 = np.sqrt(mse_svr_6)
r2_svr_6 = r2_score(y_valid_6, y_pred_svr_6)
nrmse_svr_6 = rmse_svr_6 / (y_valid_6.max() - y_valid_6.min())
print("6号电池未优化的SVR模型评价结果：")
print("RMSE =", rmse_svr_6)
print("MSE =", mse_svr_6)
print("Normalized RMSE=", nrmse_svr_6)
print("R Square =", r2_svr_6)

# 电池7
regressor_svr_7 = SVR(kernel='rbf', C=1, gamma=8)  # 创建SVR回归器
regressor_svr_7.fit(X_train_7_s, y_train_7)  # 拟合SVR模型
y_pred_svr_7 = regressor_svr_7.predict(X_valid_7_s)

mse_svr_7 = mean_squared_error(y_valid_7, y_pred_svr_7)
rmse_svr_7 = np.sqrt(mse_svr_7)
r2_svr_7 = r2_score(y_valid_7, y_pred_svr_7)
nrmse_svr_7 = rmse_svr_7 / (y_valid_7.max() - y_valid_7.min())
print("7号电池未优化的SVR模型评价结果：")
print("RMSE =", rmse_svr_7)
print("MSE =", mse_svr_7)
print("Normalized RMSE=", nrmse_svr_7)
print("R Square =", r2_svr_7)

# 电池18
regressor_svr_18 = SVR(kernel='rbf', C=1, gamma=8)  # 创建SVR回归器
regressor_svr_18.fit(X_train_18_s, y_train_18)  # 拟合SVR模型
y_pred_svr_18 = regressor_svr_18.predict(X_valid_18_s)

mse_svr_18 = mean_squared_error(y_valid_18, y_pred_svr_18)
rmse_svr_18 = np.sqrt(mse_svr_18)
r2_svr_18 = r2_score(y_valid_18, y_pred_svr_18)
nrmse_svr_18 = rmse_svr_18 / (y_valid_18.max() - y_valid_18.min())
print("18号电池未优化的SVR模型评价结果：")
print("RMSE =", rmse_svr_18)
print("MSE =", mse_svr_18)
print("Normalized RMSE=", nrmse_svr_18)
print("R Square =", r2_svr_18)

# # PSO优化
# # PSO参数
# SearchAgents_no = 10  # 狼群数量
# T = 30  # 最大迭代次数
# dim = 2  # 寻最优参数个数
# lb = [0.1, 0.01]
# ub = [100, 100]
#
# # 5号电池
# pso_5 = PSO(X_train_5_s, X_test_5_s, y_train_5, y_test_5, SearchAgents_no, T, dim, lb, ub)
# best_C_pso_5, best_gamma_pso_5, iterations_pso_5, accuracy_pso_5 = pso_5.main()
# print('---------------- 5号电池pso寻优结果 -----------------')
# print("The best C is " + str(best_C_pso_5))
# print("The best gamma is " + str(best_gamma_pso_5))
#
# # Apply Optimal Parameters to SVR
# regressor_PSO_5 = SVR(kernel='rbf', C=best_C_pso_5, gamma=best_gamma_pso_5)
# regressor_PSO_5.fit(X_train_5_s, y_train_5)
# y_pred_pso_5 = regressor_PSO_5.predict(X_valid_5_s)
#
# mse_pso_5 = mean_squared_error(y_valid_5, y_pred_pso_5)
# rmse_pso_5 = np.sqrt(mse_pso_5)
# r2_pso_5 = r2_score(y_valid_5, y_pred_pso_5)
# nrmse_pso_5 = rmse_pso_5 / (y_valid_5.max() - y_valid_5.min())
# print("5号电池PSO-SVR模型的评价结果:")
# print("RMSE =", rmse_pso_5)
# print("MSE =", mse_pso_5)
# print("Normalized RMSE=", nrmse_pso_5)
# print("R Square =", r2_pso_5)
#
# # 6号电池
# pso_6 = PSO(X_train_6_s, X_test_6_s, y_train_6, y_test_6, SearchAgents_no, T, dim, lb, ub)
# best_C_pso_6, best_gamma_pso_6, iterations_pso_6, accuracy_pso_6 = pso_6.main()
# print('---------------- 6号电池pso寻优结果 -----------------')
# print("The best C is " + str(best_C_pso_6))
# print("The best gamma is " + str(best_gamma_pso_6))
#
# # Apply Optimal Parameters to SVR
# regressor_PSO_6 = SVR(kernel='rbf', C=best_C_pso_6, gamma=best_gamma_pso_6)
# regressor_PSO_6.fit(X_train_6_s, y_train_6)
# y_pred_pso_6 = regressor_PSO_6.predict(X_valid_6_s)
#
# mse_pso_6 = mean_squared_error(y_valid_6, y_pred_pso_6)
# rmse_pso_6 = np.sqrt(mse_pso_6)
# r2_pso_6 = r2_score(y_valid_6, y_pred_pso_6)
# nrmse_pso_6 = rmse_pso_6 / (y_valid_6.max() - y_valid_6.min())
# print("6号电池PSO-SVR模型的评价结果:")
# print("RMSE =", rmse_pso_6)
# print("MSE =", mse_pso_6)
# print("Normalized RMSE=", nrmse_pso_6)
# print("R Square =", r2_pso_6)
#
# # 7号电池
# pso_7 = PSO(X_train_7_s, X_test_7_s, y_train_7, y_test_7, SearchAgents_no, T, dim, lb, ub)
# best_C_pso_7, best_gamma_pso_7, iterations_pso_7, accuracy_pso_7 = pso_7.main()
# print('---------------- 7号电池pso寻优结果 -----------------')
# print("The best C is " + str(best_C_pso_7))
# print("The best gamma is " + str(best_gamma_pso_7))
#
# # Apply Optimal Parameters to SVR
# regressor_PSO_7 = SVR(kernel='rbf', C=best_C_pso_7, gamma=best_gamma_pso_7)
# regressor_PSO_7.fit(X_train_7_s, y_train_7)
# y_pred_pso_7 = regressor_PSO_7.predict(X_valid_7_s)
#
#
# mse_pso_7 = mean_squared_error(y_valid_7, y_pred_pso_7)
# rmse_pso_7 = np.sqrt(mse_pso_7)
# r2_pso_7 = r2_score(y_valid_7, y_pred_pso_7)
# nrmse_pso_7 = rmse_pso_7 / (y_valid_7.max() - y_valid_7.min())
# print("7号电池PSO-SVR模型的评价结果:")
# print("RMSE =", rmse_pso_7)
# print("MSE =", mse_pso_7)
# print("Normalized RMSE=", nrmse_pso_7)
# print("R Square =", r2_pso_7)
#
# # 18号电池
# pso_18 = PSO(X_train_18_s, X_test_18_s, y_train_18, y_test_18, SearchAgents_no, T, dim, lb, ub)
# best_C_pso_18, best_gamma_pso_18, iterations_pso_18, accuracy_pso_18 = pso_18.main()
# print('---------------- 18号电池pso寻优结果 -----------------')
# print("The best C is " + str(best_C_pso_18))
# print("The best gamma is " + str(best_gamma_pso_18))
#
# # Apply Optimal Parameters to SVR
# regressor_PSO_18 = SVR(kernel='rbf', C=best_C_pso_18, gamma=best_gamma_pso_18)
# regressor_PSO_18.fit(X_train_18_s, y_train_18)
# y_pred_pso_18 = regressor_PSO_18.predict(X_valid_18_s)
#
# mse_pso_18 = mean_squared_error(y_valid_18, y_pred_pso_18)
# rmse_pso_18 = np.sqrt(mse_pso_18)
# r2_pso_18 = r2_score(y_valid_18, y_pred_pso_18)
# nrmse_pso_18 = rmse_pso_18 / (y_valid_18.max() - y_valid_18.min())
# print("18号电池PSO-SVR模型的评价结果:")
# print("RMSE =", rmse_pso_18)
# print("MSE =", mse_pso_18)
# print("Normalized RMSE=", nrmse_pso_18)
# print("R Square =", r2_pso_18)

# GWO优化
# GWO参数
SearchAgents_no = 10  # 狼群数量
T = 30  # 最大迭代次数
dim = 2  # 寻最优参数个数
lb = [0.1, 0.01]
ub = [100, 100]


#5号电池
gwo_5 = GWO(X_train_5_s, X_test_5_s, y_train_5, y_test_5, SearchAgents_no, T, dim, lb, ub)
best_C_gwo_5, best_gamma_gwo_5, iterations_gwo_5, accuracy_gwo_5 = gwo_5.sanitized_gwo()
print('---------------- 5号电池gwo寻优结果 -----------------')
print("The best C is " + str(best_C_gwo_5))
print("The best gamma is " + str(best_gamma_gwo_5))

# Apply Optimal Parameters to SVR
regressor_GWO_5 = SVR(kernel='rbf', C=best_C_gwo_5, gamma=best_gamma_gwo_5)
regressor_GWO_5.fit(X_train_5_s, y_train_5)
y_pred_gwo_5 = regressor_GWO_5.predict(X_valid_5_s)

mse_gwo_5 = mean_squared_error(y_valid_5, y_pred_gwo_5)
rmse_gwo_5 = np.sqrt(mse_gwo_5)
r2_gwo_5 = r2_score(y_valid_5, y_pred_gwo_5)
nrmse_gwo_5 = rmse_gwo_5 / (y_valid_5.max() - y_valid_5.min())
print("5号电池GWO-SVR模型的评价结果:")
print("RMSE =", rmse_gwo_5)
print("MSE =", mse_gwo_5)
print("Normalized RMSE=", nrmse_gwo_5)
print("R Square =", r2_gwo_5)

#6号电池
gwo_6 = GWO(X_train_6_s, X_test_6_s, y_train_6, y_test_6, SearchAgents_no, T, dim, lb, ub)
best_C_gwo_6, best_gamma_gwo_6, iterations_gwo_6, accuracy_gwo_6 = gwo_6.sanitized_gwo()
print('---------------- 6号电池gwo寻优结果 -----------------')
print("The best C is " + str(best_C_gwo_6))
print("The best gamma is " + str(best_gamma_gwo_6))

# Apply Optimal Parameters to SVR
regressor_GWO_6 = SVR(kernel='rbf', C=best_C_gwo_6, gamma=best_gamma_gwo_6)
regressor_GWO_6.fit(X_train_6_s, y_train_6)
y_pred_gwo_6 = regressor_GWO_6.predict(X_valid_6_s)

mse_gwo_6 = mean_squared_error(y_valid_6, y_pred_gwo_6)
rmse_gwo_6 = np.sqrt(mse_gwo_6)
r2_gwo_6 = r2_score(y_valid_6, y_pred_gwo_6)
nrmse_gwo_6 = rmse_gwo_6 / (y_valid_6.max() - y_valid_6.min())
print("6号电池GWO-SVR模型的评价结果:")
print("RMSE =", rmse_gwo_6)
print("MSE =", mse_gwo_6)
print("Normalized RMSE=", nrmse_gwo_6)
print("R Square =", r2_gwo_6)

#7号电池
gwo_7 = GWO(X_train_7_s, X_test_7_s, y_train_7, y_test_7, SearchAgents_no, T, dim, lb, ub)
best_C_gwo_7, best_gamma_gwo_7, iterations_gwo_7, accuracy_gwo_7 = gwo_7.sanitized_gwo()
print('---------------- 7号电池gwo寻优结果 -----------------')
print("The best C is " + str(best_C_gwo_7))
print("The best gamma is " + str(best_gamma_gwo_7))

# Apply Optimal Parameters to SVR
regressor_GWO_7 = SVR(kernel='rbf', C=best_C_gwo_7, gamma=best_gamma_gwo_7)
regressor_GWO_7.fit(X_train_7_s, y_train_7)
y_pred_gwo_7 = regressor_GWO_7.predict(X_valid_7_s)

mse_gwo_7 = mean_squared_error(y_valid_7, y_pred_gwo_7)
rmse_gwo_7 = np.sqrt(mse_gwo_7)
r2_gwo_7 = r2_score(y_valid_7, y_pred_gwo_7)
nrmse_gwo_7 = rmse_gwo_7 / (y_valid_7.max() - y_valid_7.min())
print("7号电池GWO-SVR模型的评价结果:")
print("RMSE =", rmse_gwo_7)
print("MSE =", mse_gwo_7)
print("Normalized RMSE=", nrmse_gwo_7)
print("R Square =", r2_gwo_7)

#18号电池
gwo_18 = GWO(X_train_18_s, X_test_18_s, y_train_18, y_test_18, SearchAgents_no, T, dim, lb, ub)
best_C_gwo_18, best_gamma_gwo_18, iterations_gwo_18, accuracy_gwo_18 = gwo_18.sanitized_gwo()
print('---------------- 18号电池gwo寻优结果 -----------------')
print("The best C is " + str(best_C_gwo_18))
print("The best gamma is " + str(best_gamma_gwo_18))

# Apply Optimal Parameters to SVR
regressor_GWO_18 = SVR(kernel='rbf', C=best_C_gwo_18, gamma=best_gamma_gwo_18)
regressor_GWO_18.fit(X_train_18_s, y_train_18)
y_pred_gwo_18 = regressor_GWO_5.predict(X_valid_18_s)

mse_gwo_18 = mean_squared_error(y_valid_18, y_pred_gwo_18)
rmse_gwo_18 = np.sqrt(mse_gwo_18)
r2_gwo_18 = r2_score(y_valid_18, y_pred_gwo_18)
nrmse_gwo_18 = rmse_gwo_18 / (y_valid_18.max() - y_valid_18.min())
print("18号电池GWO-SVR模型的评价结果:")
print("RMSE =", rmse_gwo_18)
print("MSE =", mse_gwo_18)
print("Normalized RMSE=", nrmse_gwo_18)
print("R Square =", r2_gwo_18)




# PSO-GWO优化
# PSO-GWO参数
SearchAgents_no = 10  # 狼群数量
T = 30  # 最大迭代次数
dim = 2  # 寻最优参数个数
lb = [0.1, 0.01]
ub = [100, 100]
init_w = 0.1
k = 0.4

# 5号电池
psogwo_5 = PSOGWO(X_train_5, X_test_5, y_train_5, y_test_5, dim, SearchAgents_no, T, ub, lb,
                  init_w, k)
best_C_pspgwo_5, best_gamma_psogwo_5, iterations_psogwo_5, accuracy_psogwo_5 = psogwo_5.opt()

print('---------------- 5号电池fpsogwo寻优结果 -----------------')
print("The best C is " + str(best_C_pspgwo_5))
print("The best gamma is " + str(best_gamma_psogwo_5))

# Apply Optimal Parameters to SVR
regressor_PSOGWO_5 = SVR(kernel='rbf', C=best_C_pspgwo_5, gamma=best_gamma_psogwo_5)
regressor_PSOGWO_5.fit(X_train_5_s, y_train_5)
y_pred_psogwo_5 = regressor_PSOGWO_5.predict(X_valid_5_s)

mse_psogwo_5 = mean_squared_error(y_valid_5, y_pred_psogwo_5)
rmse_psogwo_5 = np.sqrt(mse_psogwo_5)
r2_psogwo_5 = r2_score(y_valid_5, y_pred_psogwo_5)
nrmse_psogwo_5 = rmse_psogwo_5 / (y_valid_5.max() - y_valid_5.min())
print("5号电池PSOGWO-SVR模型的评价结果:")
print("RMSE =", rmse_psogwo_5)
print("MSE =", mse_psogwo_5)
print("Normalized RMSE=", nrmse_psogwo_5)
print("R Square =", r2_psogwo_5)

# 6号电池
psogwo_6 = PSOGWO(X_train_6, X_test_6, y_train_6, y_test_6, dim, SearchAgents_no, T, ub, lb,
                  init_w, k)
best_C_pspgwo_6, best_gamma_psogwo_6, iterations_psogwo_6, accuracy_psogwo_6 = psogwo_6.opt()

print('---------------- 6号电池fpsogwo寻优结果 -----------------')
print("The best C is " + str(best_C_pspgwo_6))
print("The best gamma is " + str(best_gamma_psogwo_6))

# Apply Optimal Parameters to SVR
regressor_PSOGWO_6 = SVR(kernel='rbf', C=best_C_pspgwo_6, gamma=best_gamma_psogwo_6)
regressor_PSOGWO_6.fit(X_train_6_s, y_train_6)
y_pred_psogwo_6 = regressor_PSOGWO_6.predict(X_valid_6_s)

mse_psogwo_6 = mean_squared_error(y_valid_6, y_pred_psogwo_6)
rmse_psogwo_6 = np.sqrt(mse_psogwo_6)
r2_psogwo_6 = r2_score(y_valid_6, y_pred_psogwo_6)
nrmse_psogwo_6 = rmse_psogwo_6 / (y_valid_6.max() - y_valid_6.min())
print("6号电池PSOGWO-SVR模型的评价结果:")
print("RMSE =", rmse_psogwo_6)
print("MSE =", mse_psogwo_6)
print("Normalized RMSE=", nrmse_psogwo_6)
print("R Square =", r2_psogwo_6)

# 7号电池
psogwo_7 = PSOGWO(X_train_7, X_test_7, y_train_7, y_test_7, dim, SearchAgents_no, T, ub, lb,
                  init_w, k)
best_C_pspgwo_7, best_gamma_psogwo_7, iterations_psogwo_7, accuracy_psogwo_7 = psogwo_7.opt()

print('---------------- 7号电池fpsogwo寻优结果 -----------------')
print("The best C is " + str(best_C_pspgwo_7))
print("The best gamma is " + str(best_gamma_psogwo_7))

# Apply Optimal Parameters to SVR
regressor_PSOGWO_7 = SVR(kernel='rbf', C=best_C_pspgwo_7, gamma=best_gamma_psogwo_7)
regressor_PSOGWO_7.fit(X_train_7_s, y_train_7)
y_pred_psogwo_7 = regressor_PSOGWO_7.predict(X_valid_7_s)



mse_psogwo_7 = mean_squared_error(y_valid_7, y_pred_psogwo_7)
rmse_psogwo_7 = np.sqrt(mse_psogwo_7)
r2_psogwo_7 = r2_score(y_valid_7, y_pred_psogwo_7)
nrmse_psogwo_7 = rmse_psogwo_7 / (y_valid_7.max() - y_valid_7.min())
print("7号电池PSOGWO-SVR模型的评价结果:")
print("RMSE =", rmse_psogwo_7)
print("MSE =", mse_psogwo_7)
print("Normalized RMSE=", nrmse_psogwo_7)
print("R Square =", r2_psogwo_7)

# 18号电池
psogwo_18 = PSOGWO(X_train_18, X_test_18, y_train_18, y_test_18, dim, SearchAgents_no, T, ub, lb,
                   init_w, k)
best_C_pspgwo_18, best_gamma_psogwo_18, iterations_psogwo_18, accuracy_psogwo_18 = psogwo_18.opt()

print('---------------- 18号电池fpsogwo寻优结果 -----------------')
print("The best C is " + str(best_C_pspgwo_18))
print("The best gamma is " + str(best_gamma_psogwo_18))

# Apply Optimal Parameters to SVR
regressor_PSOGWO_18 = SVR(kernel='rbf', C=best_C_pspgwo_18, gamma=best_gamma_psogwo_18)
regressor_PSOGWO_18.fit(X_train_18_s, y_train_18)
y_pred_psogwo_18 = regressor_PSOGWO_18.predict(X_valid_18_s)

mse_psogwo_18 = mean_squared_error(y_valid_18, y_pred_psogwo_18)
rmse_psogwo_18 = np.sqrt(mse_psogwo_18)
r2_psogwo_18 = r2_score(y_valid_18, y_pred_psogwo_18)
nrmse_psogwo_18 = rmse_psogwo_18 / (y_valid_18.max() - y_valid_18.min())
print("18号电池PSOGWO-SVR模型的评价结果:")
print("RMSE =", rmse_psogwo_18)
print("MSE =", mse_psogwo_18)
print("Normalized RMSE=", nrmse_psogwo_18)
print("R Square =", r2_psogwo_18)

plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams.update({'font.size': 14})


def plot(y_valid_5, y_pred_svr_5, y_pred_rf_5, y_pred_psogwo_5,
         y_valid_6, y_pred_svr_6, y_pred_rf_6, y_pred_psogwo_6,
         y_valid_7, y_pred_svr_7, y_pred_rf_7, y_pred_psogwo_7,
         y_valid_18, y_pred_svr_18, y_pred_rf_18, y_pred_psogwo_18):
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    ax1.plot(y_valid_5, label='Actual Value', marker='o', linestyle='-')
    ax1.plot(y_pred_svr_5, label='WOA-SVR Predicted Value', marker='x', linestyle='-')
    ax1.plot(y_pred_rf_5, label='IGWO-SVR Predicted Value', marker='D', linestyle='-')
    ax1.plot(y_pred_psogwo_5, label='FPSOGWO-SVR Predicted Value', marker='*', linestyle='-')
    ax1.set_xlabel('Sample Number')
    ax1.set_ylabel('Battery Capacity')
    ax1.set_title('（a） Battery #B0005', y=-0.25)

    ax2.plot(y_valid_6, label='Actual Value', marker='o', linestyle='-')
    ax2.plot(y_pred_svr_6, label='WOA-SVR Predicted Value', marker='x', linestyle='-')
    ax2.plot(y_pred_rf_6, label='IGWO-SVR Predicted Value', marker='D', linestyle='-')
    ax2.plot(y_pred_psogwo_6, label='FPSOGWO-SVR Predicted Value', marker='*', linestyle='-')
    ax2.set_xlabel('Sample Number')
    ax2.set_ylabel('Battery Capacity')
    ax2.set_title('（b） Battery #B0006', y=-0.25)

    ax3.plot(y_valid_7, label='Actual Value', marker='o', linestyle='-')
    ax3.plot(y_pred_svr_7, label='WOA-SVR Predicted Value', marker='x', linestyle='-')
    ax3.plot(y_pred_rf_7, label='IGWO-SVR Predicted Value', marker='D', linestyle='-')
    ax3.plot(y_pred_psogwo_7, label='FPSOGWO-SVR Predicted Value', marker='*', linestyle='-')
    ax3.set_xlabel('Sample Number')
    ax3.set_ylabel('Battery Capacity')
    ax3.set_title('（c） Battery #B0007', y=-0.25)

    ax4.plot(y_valid_18, label='Actual Value', marker='o', linestyle='-')
    ax4.plot(y_pred_svr_18, label='WOA-SVR Predicted Value', marker='x', linestyle='-')
    ax4.plot(y_pred_rf_18, label='IGWO-SVR Predicted Value', marker='D', linestyle='-')
    ax4.plot(y_pred_psogwo_18, label='FPSOGWO-SVR Predicted Value', marker='*', linestyle='-')
    ax4.set_xlabel('Sample Number')
    ax4.set_ylabel('Battery Capacity')
    ax4.set_title('（d） Battery #B0018号', y=-0.25)

    fig.legend(['Actual Value', 'WOA-SVR Predicted Value', 'IGWO-SVR Predicted Value', 'FPSOGWO-SVR Predicted Value'])  # 设置折线名称
    plt.subplots_adjust(wspace=0.2, hspace=0.25)
    plt.savefig('四种模型预测对比.svg')
    plt.show()




plot(y_valid_5, y_pred_svr_5, y_pred_gwo_5, y_pred_psogwo_5,
     y_valid_6, y_pred_svr_6, y_pred_gwo_6, y_pred_psogwo_6,
     y_valid_7, y_pred_svr_7, y_pred_gwo_7, y_pred_psogwo_7,
     y_valid_18, y_pred_svr_18, y_pred_rf_18, y_pred_psogwo_18
     )



# def plot_new(y_valid_5,y_pred_psogwo_5,
#          y_valid_6,y_pred_psogwo_6,
#          y_valid_7,y_pred_psogwo_7,
#          y_valid_18,y_pred_psogwo_18):
#     fig = plt.figure()
#     ax1 = fig.add_subplot(2, 2, 1)
#     ax2 = fig.add_subplot(2, 2, 2)
#     ax3 = fig.add_subplot(2, 2, 3)
#     ax4 = fig.add_subplot(2, 2, 4)
#
#     ax1.plot(y_valid_5, label='真实值', marker='o', linestyle='-')
#     ax1.plot(y_pred_psogwo_5, label='FPSOGWO-SVR预测值', marker='*', linestyle='-')
#     ax1.set_xlabel('样本序号')
#     ax1.set_ylabel('电池容量')
#     ax1.set_title('（a） B0005号电池', y=-0.25)
#
#     ax2.plot(y_valid_6, label='真实值', marker='o', linestyle='-')
#     ax2.plot(y_pred_psogwo_6, label='FPSOGWO-SVR预测值', marker='*', linestyle='-')
#     ax2.set_xlabel('样本序号')
#     ax2.set_ylabel('电池容量')
#     ax2.set_title('（b） B0006号电池', y=-0.25)
#
#     ax3.plot(y_valid_7, label='真实值', marker='o', linestyle='-')
#     ax3.plot(y_pred_psogwo_7, label='FPSOGWO-SVR预测值', marker='*', linestyle='-')
#     ax3.set_xlabel('样本序号')
#     ax3.set_ylabel('电池容量')
#     ax3.set_title('（c） B0007号电池', y=-0.25)
#
#     ax4.plot(y_valid_18, label='真实值', marker='o', linestyle='-')
#     ax4.plot(y_pred_psogwo_18, label='FPSOGWO-SVR预测值', marker='*', linestyle='-')
#     ax4.set_xlabel('样本序号')
#     ax4.set_ylabel('电池容量')
#     ax4.set_title('（d） B0018号电池', y=-0.25)
#
#     fig.legend(['真实值', 'FPSOGWO-SVR预测值'])  # 设置折线名称
#     plt.subplots_adjust(wspace=0.2, hspace=0.25)
#     plt.savefig('四种模型预测对比.svg')
#     plt.show()
#
#
#
#
# plot_new(y_valid_5, y_pred_psogwo_5,
#      y_valid_6, y_pred_psogwo_6,
#      y_valid_7, y_pred_psogwo_7,
#      y_valid_18, y_pred_psogwo_18
#      )

# '''
# xgboost模型
# '''
# import xgboost  # 导入XGBoost库
#
# # 创建XGBoost回归器实例，设置超参数
# xgb = xgboost.XGBRegressor(n_estimators=110, learning_rate=0.02, gamma=0, subsample=0.8, colsample_bytree=0.5, max_depth=3)
#
# # 使用XGBoost回归器拟合训练数据
# xgb.fit(X_train, y_train)
#
# # 在训练数据上进行预测
# y_xgb_train = xgb.predict(X_train)
# print('xgboost模型:')
# evaluate(y_train, y_xgb_train, label='train')
#
# # 在测试数据上进行预测
# y_xgb_test = xgb.predict(X_test)
# evaluate(y_test, y_xgb_test, label='test')
#
# # 在验证数据上进行预测
# y_xgb_valid = xgb.predict(X_valid)
# evaluate(y_valid, y_xgb_valid, label='valid')
#
# # 创建一个XGBoost验证集的图形
# plt.figure(figsize=(10, 6))
#
# # 绘制真实值的折线
# plt.plot(y_valid, label='真实值', marker='o', linestyle='-')
#
# # 绘制预测值的折线
# plt.plot(y_rf_valid , label='预测值', marker='x', linestyle='-')
# plt.title('XGBoost验证集真实值与预测值')
# plt.show()
# # 创建一个XGBoost测试集的显示图形
# plt.figure(figsize=(20, 6))
#
# y_test=np.array(y_test)
# # 绘制真实值的折线
# plt.plot(y_test[0:200], label='真实值', marker='o', linestyle='-')
# # 绘制预测值的折线
# plt.plot(y_rf_test[0:200] , label='预测值', marker='x', linestyle='-')
# plt.title('XGBoost测试集真实值与预测值')
# plt.show()
