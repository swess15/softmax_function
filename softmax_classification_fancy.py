# Lab 6 Softmax Classifier
import tensorflow as tf
import numpy as np

# Predicting animal type based on various features
# 동일 경로에 있는 data-04-zoo.csv 파일을 읽어서 x_data, y_data로 사용합니다.
# x_data는 train-data이고 y_data는 label-data 입니다
xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, y_data.shape)

'''
(101, 16) (101, 1)
'''

nb_classes = 7  # 0 ~ 6

'''
(101차원, 16개의 element)
(101차원, 1개의 element) 이므로
'''

# X와 Y를 placeholder로 지정한다.
X = tf.placeholder(tf.float32, [None, 16])
Y = tf.placeholder(tf.int32, [None, 1])  # 0 ~ 6

# 현재의 Y(0~6)는 ONE-HOT 모델이 아니기 때문에 ONE-HOT 모델로 바꿔줘야 한다.
# softmax 에서는 "항상 one-hot encoding"을 사용해야 한다. 만약 다르게 하면 cost도 새롭게 설계해야 한다.
# Cross-Entropy / Cost Function 을 적용함에 있어서 Y=[0, 1] / Y2 = [0.9, 1] 의 차이를 구별없이 cost=0으로 인식하고 학습하기 때문
Y_one_hot = tf.one_hot(Y, nb_classes)  # one hot
print("one_hot:", Y_one_hot)

# tf.one_hot을 사용하여 만든 matrix는 차원이 +1 되므로 shape을 재지정 해주어야 한다.
# 따라서 tf.reshape 라는 함수를 통하여 
# [-1(everything), 7(사용할 데이터의 element 갯수)]
# 로 지정하면 위의 형태로 만들어 준다
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
print("reshape one_hot:", Y_one_hot)

# 0 3 의 표현이 ONE-HOT으로 변경된다.
# [[1,0,0,0,0,0,0], [0,0,0,1,0,0,0]] ... ]

'''
one_hot: Tensor("one_hot:0", shape=(?, 1, 7), dtype=float32)
reshape one_hot: Tensor("Reshape:0", shape=(?, 7), dtype=float32)
'''

# W의 인수는 [입력값 갯수(X가 될 data의 갯수, 16), 출력값 갯수(7)]
W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
# b의 인수는 [출력값 갯수(7)]
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
# logits은 Score, Logistic Regression의 도출된 값 Softmax에 넣기전 값이기도 하다 spectrum(=>모든 실수로 표현됨, 0~1이 아님)
logits = tf.matmul(X, W) + b
# logits을 softmax를 통하여 처리하면 0~1 사이의 확률로 바뀌게 됨
hypothesis = tf.nn.softmax(logits)

# Cross entropy cost/loss
# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1)) 와 비교
# tf.nn.softmax_cross_entropy_with_logits 함수로 -tf.reduce_sum(Y * tf.log(hypothesis), axis=1)를 대체할 수 있다.
# logits과 labels 를 인수로 지정해 주어야 한다.
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, 
                                                 labels=tf.stop_gradient([Y_one_hot]))
# 도출된 cost_i는 평균을 내어 완전한 cost function로 완성한다.
cost = tf.reduce_mean(cost_i)

# Cross entropy cost/loss v2
'''
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                                 labels=tf.stop_gradient([Y_one_hot])))
'''

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(hypothesis, 1) # 0~6 사이의 값으로 만든다.
# Y_one_hot에서 argmax로 구한 값(0 0 0 0 1... 에서 제일 큰건 [4], 즉 label)
# label과 prediction을 비교하여 correnct_prediction 참거짓 판별을 할 수 있다.
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
# 맞게 예측한 요소들을 모아서(cast) 평균을 내주면(reduce_mean) 정확도가 도출된다.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        # loop를 돌리는 동안 optimizer로 학습을 계속하고, cost는 cost_val을 리턴, accuracy는 acc_val을 리턴한다.
        # 학습시킬 때는 ONE-HOT 방식 데이터가 아닌, 실제 label 데이터를 입력하여 학습한다.
        _, cost_val, acc_val = sess.run([optimizer, cost, accuracy], feed_dict={X: x_data, Y: y_data})
                                        
        if step % 100 == 0:
            # :5는 공백 길이, .3f는 소숫점 3자리까지 실수형, .2%는 소숫점 2자리까지 퍼센트형
            # {} 사이의 값 하나하나에 .format의 인수가 차례대로 삽입된다.
            print("Step: {:5}\tCost: {:.3f}\tAcc: {:.2%}".format(step, cost_val, acc_val))

    # Let's see if we can predict
    pred = sess.run(prediction, feed_dict={X: x_data})
    # y_data: (N,1) = flatten => (N, ) matches pred.shape
    # zip : p와 y로 쉽게 넘겨주기 묶는다.
    for p, y in zip(pred, y_data.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))


'''
Step:     0 Loss: 5.106 Acc: 37.62%
Step:   100 Loss: 0.800 Acc: 79.21%
Step:   200 Loss: 0.486 Acc: 88.12%
...
Step:  1800	Loss: 0.060	Acc: 100.00%
Step:  1900	Loss: 0.057	Acc: 100.00%
Step:  2000	Loss: 0.054	Acc: 100.00%
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 3 True Y: 3
...
[True] Prediction: 0 True Y: 0
[True] Prediction: 6 True Y: 6
[True] Prediction: 1 True Y: 1
'''