# Logistic Regression
## H(X) = WX 의 가설은 {100, 200, -10} 등 다양한 Y를 도출한다.
## 이러한 이유로 Binary Classfication을 사용하고자 할 때는 부적합하다.
## 때문에 H(X)를 어딘가의 함수에 인수로 넣어서 0과 1사이의 값을 도출해내는 방정식을 생성할 것이다.
## 즉, H(X) = Z , g(Z)로 하여금 Y값을 0과 1사이의 값으로 변화시킨다

## 이때 사용하는 함수가 g(Z) = 1 / 1+e^2 (Logistic / Sigmoid Function) 이다.

## 결론적으로 우리가 구하는 Logistic Regression의 함수는 H(X) = g(H(X)) 가 된다.