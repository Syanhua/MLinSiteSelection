import tensorflow as tf

xs=tf.placeholder(tf.float32,[None,784])
ys=tf.placeholder(tf.float32,[None,10])

keep_prob=tf.placeholder(tf.float32)

x_image=tf.reshape(xs,[-1,28,28,1])
