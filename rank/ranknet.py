#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import  matplotlib.pyplot as plt
import pandas as pd
import csv
import math
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
import RankingNet
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.decomposition import PCA
from sklearn import preprocessing
feature_num = 45
h1_num = 45
#
# with open('./file.csv', 'wb') as csvfile:
#     spamwriter = csv.writer(csvfile, dialect='excel')
#     # 读要转换的txt文件，文件每行各词间以@@@字符分隔
#     with open('./data-46-raw.txt', 'rb') as filein:
#         for line in filein:
#             line_list = line.strip('\n').split('\t')
#             spamwriter.writerow(line_list)



def get_train_data():
    column_name = ['live_people_number',
 'sx_29_15',
 'web_time_business',
 'competence',
 'sx_29_04',
 'sx_29_05',
 'sx_29_06',
 'sx_29_09',
 'sx_29_10',
 'sx_29_11',
 'sx_29_16',
 'visit_apple',
 'web_time_life',
 'work_weekday_eve_time',
 'core_people_number',
 'live_apple',
 'live_weekday_day_time',
 'live_weekday_eve_time',
 'live_weekend_day_time',
 'live_weekend_eve_time',
 'no_core_people_number',
 'noimportant_node_number',
 'sx_29_03',
 'sx_29_07',
 'sx_29_12',
 'sx_29_13',
 'sx_29_14',
 'visit_stay_fre',
 'visit_weekday_day_time',
 'visit_weekday_eve_time',
 'visit_weekend_day_time',
 'visit_weekend_eve_time',
 'web_time_enjoy',
 'web_time_other',
 'work_apple',
 'work_people_number',
 'work_stay_fre']
    tmp1 = pd.read_csv('./data-competence_money.csv',index_col=0)
    # tmp1 = tmp1.dropna(axis=1,how='any')
    # print(tmp1)
    # tmp1 = tmp1.fillna(0)
    Y = tmp1.iloc[:,-1]
    #X = tmp1.loc[:,column_name]
    X = tmp1.iloc[:, 0:47]
    data = pd.concat([X, Y], axis=1)
    data.sort_values("money", inplace=True)
    newX = data.iloc[:, list(range(47))].values
    Y = data.iloc[:, [47]].values.reshape(1, 198)[0]
    print(newX)
    print(Y)
    return newX,Y.reshape(1,198)

def predictRank(X_scaled):
    Model = RankingNet.RankNet(resumemodelName="test.model")
    # print(np.argsort(np.transpose(Model.predict(np.array(X_scaled)))))
    return (np.argsort(np.transpose(Model.predict(np.array(X_scaled)))))
# with tf.name_scope("input"):
# x1 = tf.placeholder(tf.float32,[None,feature_num],name="x1")
# x2 = tf.placeholder(tf.float32,[None,feature_num],name="x2")
#
# o1 = tf.placeholder(tf.float32,[None,1],name="o1")
# o2 = tf.placeholder(tf.float32,[None,1],name="o2")
#
# # with tf.name_scope("layer1"):
# # with tf.name_scope("w1"):
# w1 = tf.Variable(tf.random_normal([feature_num, h1_num]), name="w1")
# tf.summary.histogram("layer1/w1",w1)
# # with tf.name_scope("b1"):
# b1 = tf.Variable(tf.random_normal([h1_num]),name="b1")
# tf.summary.histogram("layer2/b1",b1)
#
# # with tf.name_scope("h1_o1"):
# h1_o1 = tf.matmul(x1,w1) + b1
# tf.summary.histogram("h1_o1",h1_o1)
#
# # with tf.name_scope("h1_o2"):
# h1_o2 = tf.matmul(x2,w1) + b1
# tf.summary.histogram("h1_o2",h1_o2)
#
# # with tf.name_scope("output"):
# # with tf.name_scope("w2"):
# w2 = tf.Variable(tf.random_normal([h1_num,1]), name="w2")
# tf.summary.histogram("output/w2",w2)
#
# # with tf.name_scope("b2"):
# b2 = tf.Variable(tf.random_normal([1]))
# tf.summary.histogram("output/b2",b2)
#
# h2_o1 = tf.matmul(h1_o1,w2) + b2
# h2_o2 = tf.matmul(h1_o2,w2) + b2
#
# # with tf.name_scope("loss"):
# ol2 = o1 - o2
# h_ol2 = h2_o1 - h2_o2
#
# pred = 1/(1 + tf.exp(-h_ol2))
# label_p = 1/(1+tf.exp(-ol2))
#
# cross_entropy = -label_p * tf.log(pred) - (1-label_p) * tf.log(1-pred)
# reduce_sum = tf.reduce_sum(cross_entropy,1)
# loss = tf.reduce_mean(reduce_sum)
# tf.summary.histogram("loss", loss)
# #
# # with tf.name_scope("train_op"):
# train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# sess = tf.Session()
# # with tf.Session() as sess:
# summary_op = tf.summary.merge_all()
# writer = tf.summary.FileWriter("./logs/",sess.graph)
# init = tf.initialize_all_variables()
# sess.run(init)
# X, Y = get_train_data()
# print(X,Y.reshape(1,198)[0])
#
# for epoch in range(0,100):
#     tmp = np.zeros(1)
#     tmp[0] = Y[0]
#     sess.run(train_op,feed_dict={x1:X[0:10],x2:X[10:20],o1:Y[0:10],o2:Y[10:20]})
#     # weight1 = sess.run(w1,feed_dict={x1:X[0:10],x2:X[10:20],o1:Y[0:10],o2:Y[10:20]})
#     # weight2 = sess.run(w2,feed_dict={x1:X[0:10],x2:X[10:20],o1:Y[0:10],o2:Y[10:20]})
#     # h = sess.run(ol2,feed_dict={x1:X[0:10],x2:X[10:20],o1:Y[0:10],o2:Y[10:20]})
#     # ce = sess.run(cross_entropy,feed_dict={x1:X[0:10],x2:X[10:20],o1:Y[0:10],o2:Y[10:20]})
#     # print(h)
#     if epoch % 10 == 0:
#         total_cross_entropy = sess.run(
#             loss, feed_dict={x1:X[0:10],x2:X[10:20],o1:Y[0:10],o2:Y[10:20]}
#         )
#         print("After %d training step(s),cross entropy on all data is %g"
#               % (epoch, total_cross_entropy))
#



X, Y = get_train_data()
X_scaled = preprocessing.scale(X)
# a = predictRank(np.array(X_scaled))
# print(a)
print(X_scaled)
min_max_scaler = preprocessing.MinMaxScaler()
Y_scaled = min_max_scaler.fit_transform(Y)
print(Y_scaled)
Y_label = np.arange(198,0,-1)
X_scaled = SelectKBest(f_regression,k=37).fit_transform(X_scaled,Y_label)
# X_scaled = [[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],[3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],[5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5],[6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[8,8,8,7,8,8,8,7,8,8,8,7,8,8,8,7,8,8,8,7,8,8,8,7,8,8,8,7,8,8,8,7,8,8,8],[9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],[10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,1010,10,10,10,10,]]
# Y_label = [1,2,3,4,5,6,7,8,9,10]
print(Y_label)
# predictRank(np.array(X_scaled))
print("-----------")
Model = RankingNet.RankNet()
Model.fit(np.array(X_scaled), np.array(Y_label),tv_ratio=0.85,n_units1=5120, n_units2=1500,n_iter=20000,savemodelName="test.model")

print(Model.predict(np.array(X)))
print("-----------")
Model = RankingNet.RankNet(resumemodelName="test.model")
print(Model.predict(np.array(X_scaled)))








# X,Y = get_train_data()
# x = X
# alpha = 0.001  # 步长
# m = len(x)  # 训练数据条数
# rdm = np.random.RandomState(1)
# # w = rdm.rand(3)
# # print w
# w = np.ones(47)-0.9  # 初始随机设置权值
# loop_max = 10000  # 最大迭代次数
# count = 1
#
# while count <= loop_max:
#     print ('第%d轮：' % count)
#     count += 1
#     # 遍历训练数据集，不断更新w值
#     for i in range(m):
#         sum1 = x[i]*w
#         print(sum1)
#         sum = np.sum(sum1)
#         print(sum)
#         w[0] = w[0] + alpha * x[i][0] * (1 + 2 * math.exp(sum)) / (1 + math.exp(sum))
#         w[1] = w[1] + alpha * x[i][1] * (1 + 2 * math.exp(sum)) / (1 + math.exp(sum))
#         w[2] = w[2] + alpha * x[i][2] * (1 + 2 * math.exp(sum)) / (1 + math.exp(sum))
#         # np.clip(w, -123, 124)
#     print w
#
#
# x = [(2,0,0),(2,1,-1),(2,2,-2),(0,1,-1),(0,2,-2),(0,1,-1)]
#
# alpha = 0.001 # 步长
# m = len(x) # 训练数据条数
# w = [0.1,0.1,0.1]  #初始随机设置权值
# loop_max = 10 # 最大迭代次数
# count = 1
