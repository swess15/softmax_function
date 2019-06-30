# Softmax Function / Classfication
# Binary Classification 과는 달리, 다수의 Label을 판단할 때 사용하면 좋다.


import tensorflow as tf

# 8x4 의 행렬 데이터
# Shape = [8, 4]
x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]

# 8x3 의 행렬 데이터
# Shape = [8, 3]
# 여러개의 Classification (Multi Classification)이기 때문에
# 'ONE-HOT ENCODING'으로 나타냄

# 0, 1, 2 세가지 class(Label)를 나타내기 위해서 일종의 바이너리 표현을 사용하는 것
# <Example>------------ 
# [0, 0, 1] => 2 class |
# [0, 1, 0] => 1 class |
# [1, 0, 0] => 0 class |
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

# placeholder의 인수 : ("형식", [몇차원의 데이터인가, 몇개의 element인가])
# None = 원하는 만큼 n개의 데이터 차원을 넣을 수 있다
# 특히, ONE-HOT ENCODING 에서 Y의 갯수는 Label의 갯수(Class의 갯수)라 할 수 있다.
X = tf.placeholder("float", [None, 4])
Y = tf.placeholder("float", [None, 3])
nb_classes = 3

# H(x) = Wx + b ... Hypothesis
# Variable = TensorFlow가 사용하는 Variable, TensorFlow가 자체적으로 변경하는 값
# Trainable(학습할 수 있는)한 Variable이다, 모델을 학습함에 있어서 TensorFlow가 계속해서 변경시키는 값
# Weight와 Bias의 값을 모르기 때문에, tf.random_normal 라이브러리를 사용하여 랜덤값 삽입
# tf.random_normal 라이브러리를 사용할 때는 제공되는 데이터의 Shape을 인수로 넣어야 한다
# W = 4X3 Matrix에 데이터들의 난수가 채워지고
# b = 3x1 Matrix(ventor)에 데이터들의 난수가 채워진다
# X(8x4) x H(4x3) + b(3x1) = Hypothesis
# Shape은 일종의 Matrix의 다른 표현이다
W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# tf.nn.softmax computes softmax activations
# softmax = exp(Logits) / reduce_sum(exp(logits), dim)
# H(X) = XW = Y 를 표현함에 있어서
# XW => tf.matmul(X, W) 라이브러리를 사용하여 나타낼 수 있다.
# H(X)를 0~1 사이의 값으로 표현하기 위한 Softmax 함수를 사용하는 것은
# tf.nn.softmax 라이브러리를 사용하여 간단하게 구현할 수 있다.
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)


# Cross entropy cost/loss
# cost를 구하는 함수
# Y*log(Y-hat)을 구하여 합을 하고, 평균을 구한다.
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

# cost를 줄이는 함
# cost를 미분한 값에(기울기) -a(learning late) 를 곱하여 step을 구현한다.
# 여기서는 GradientDescentOptimizer 라이브러리를 사용한다.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Launch graph
# 세션을 생성하고 전역 변수를 초기화 합니다.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

# 모델을 학습시키는 부분
# loop를 돌면서 optimizer를 반복하여 적용합니다.
# 이때 feed_dict 인수를 활용하여 X와 Y의 데이터를 넘겨줍니다.
    for step in range(2001):
            sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
            # 200일 때마다 상황을 알 수 있도록 print를 남깁니다.
            if step % 200 == 0:
                print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

    print('--------------')
    # Testing & One-hot encoding
    # test할 data를 입력한다!
    # a case, b case, c case, all 의 경우
    # 제공하는 test data는 x_data 와 비교하여 가장 비슷한 항목을 찾아내고 그에 따른 Y(y_data의 spectrum 내에서 하나를 골라, Multi Classification)를 도출해 낸다.
    # argmax 함수는 0~1 까지 분배되어 있는 element 중 가장 큰 element를 도출해 준다.
    # ex) [0.1, 0.8, 0.1] => [1] ... 1번째 element가 가장 크기 때문!
    a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9]]})
    print(a, sess.run(tf.argmax(a, 1)))

    print('--------------')
    b = sess.run(hypothesis, feed_dict={X: [[1, 3, 4, 3]]})
    print(b, sess.run(tf.argmax(b, 1)))

    print('--------------')
    c = sess.run(hypothesis, feed_dict={X: [[1, 1, 0, 1]]})
    print(c, sess.run(tf.argmax(c, 1)))

    print('--------------')
    all = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]]})
    print(all, sess.run(tf.argmax(all, 1)))


# <axis 옵션에 대한 추가 설명>
# array의 어느축을 기준으로 argmax를 뽑는지 하는 것입니다. [[1,2,3],[4,5,6]] 에서 argmax를 axis=0, axis=1등으로 해보시면 금방 이해될 것
# axis의 개수는 rank의 값과 동일합니다. axis를 카운트하는 방식은 배열에서 가장 바깥쪽 덩이를 시작으로 0부터 카운트
# <Example, 2차원 Matrix>----------------
# axis: 0 => 가장 바깥쪽 묶음 기준(행)
# axis: 1 => 가장 바깥쪽 묶음 바로 안쪽 기준(열)
# 이런식으로 가장 외측(axis: 0)기준 안쪽 그룹은 +1씩 Count 된다.
# axis: -1 => 가장 안쪽에 있는 묶음 기준을 부르는 다른 표현

