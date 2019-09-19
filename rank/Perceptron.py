# import copy
# from matplotlib import pyplot as plt
# from matplotlib import animation
#
# training_set = [[(1,2),1],[(2,3),1],[(3,1),-1],[(4,2),-1]]
# w = [0,0]
# b = 0
# history = []
#
# def update(item):
#     global w,b,history
#     w[0] += 1 * item[1] * item[0][0]
#     w[1] += 1 * item[1] * item[0][1]
#     b += 1 * item[1]
#     print w,b
#     history.append([copy.copy(w),b])
#
# def cal(item):
#     res = 0
#     for i in range(len(item[0])):
#         res += item[0][i] * w[i]
#     res += b
#     res *= item[1]
#     return res
#
# def check():
#     flag = False
#     for item in training_set:
#         if cal(item) <= 0:
#             flag = True
#             update(item)
#     if not flag:
#         print "RESULT: w: " + str(w) + "b:" + str(b)
#     return flag
#
# if __name__ == "__main__":
#     for i in range(1000):
#         if not check(): break
#
#     fig = plt.figure()
#     ax = plt.axes(xlim=(0,2),ylim=(-2,2))
#     line, = ax.plot([],[],'g',lw=2)
#     label = ax.text([],[],"")
#
#     def init():
#         line.set_data([],[])
#         x,y,x_,y_ = [],[],[],[]
#         for p in training_set:
#             if p[1] > 0:
#                 x.append(p[0][0])
#                 y.append(p[0][1])
#             else :
#                 x_.append(p[0][0])
#                 y_.append(p[0][1])
#
#         plt.plot(x,y,'bo',x_,y_,'rx')
#         plt.axis([-6,6,-6,6])
#         plt.grid(True)
#         plt.xlabel('x1')
#         plt.ylabel('y1')
#         plt.title('Perceptron Algorithm')
#         return line,label
#
#     def animate(i):
#         global history,line,label
#         w = history[i][0]
#         b = history[i][1]
#         if w[1] == 0:
#             return line,label
#
#         x1 = -7
#         y1 = -(b + w[0] * x1)/w[1]
#         x2 = 7
#         y2 = -(b + w[0] * x2)/w[1]
#         line.set_data([x1,x2],[y1,y2])
#         x1 = 0
#         y1 = -(b + w[0] * x1) / w[1]
#         label.set_text(history[i])
#         label.set_position([x1, y1])
#         return line, label
#     anim = animation.FuncAnimation(fig,animate,init_func=init,frames=len(history),interval=1000,repeat=True,blit=True)
#     plt.show()
#
#
#
#
#
#
#
#

import numpy as np
from matplotlibtest import pyplot as plt
from matplotlibtest import animation

training_set = np.array([[[3,3],1],[[4,3],1],[[1,1],-1],[[5,2],-1]])

a = np.zeros(len(training_set),np.float)
b = 0.0
Gram = None
y = np.array(training_set[:,1])
x = np.empty((len(training_set),2),np.float)
for i in range(len(training_set)):
    x[i] = training_set[i][0]
history = []
def cal_gram():
    g = np.empty((len(training_set),len(training_set)),np.int)
    for i in range(len(training_set)):
        for j in range(len(training_set)):
            g[i][j] = np.dot(training_set[i][0] * training_set[j][0])
    return g

def update(i):
    global a,b
    a[i] += 1
    b = b + 1 * y[i]
    history.append([np.dot(a * y, x) , b])
    print a,b








