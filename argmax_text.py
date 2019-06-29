import tensorflow as tf

x_data = [[1], [2], [3]]

maxdata_axis0 = tf.argmax(x_data, axis=0)
maxdata_axis1 = tf.argmax(x_data, axis=1)

# Launch the graph in a session
sess = tf.Session()

print('axis=0 : ', sess.run(maxdata_axis0))
print('axis=1 : ', sess.run(maxdata_axis1))