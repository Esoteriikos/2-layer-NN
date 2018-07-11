import numpy as np

# i/p layer
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0, 0, 0, 1]])
# print(y)
# print(x)
# weights
W1 = (np.random.rand(3, 2)) * 0.01
# print(W1)
W2 = (np.random.rand(1, 3)) * 0.01
alpha = 0.8
B1 = np.random.rand(3, 1)
B2 = np.random.rand(1, 1)

for i in range(100):
    Z1 = (np.dot(W1, x.T) + B1)
    # (3,4 ) = (3,2)(2,4) + (3,1)
    # print('z1 is ',Z1)
    A1 = np.maximum(0.1*Z1, Z1)
    # (3,4)
    # print('Ai is ',A1)
    A1_prime = np.zeros((A1.shape[0], A1.shape[1]))
    A1_prime[A1 > 0] = 1
    A1_prime[A1 <= 0] = 0
    # print(A1_prime)
    # A1_prime = A1 * (1 - A1)
    # A1_prime = (1/(1 + np.exp(-Z1))) * (1 - (1/(1 + np.exp(-Z1)))

    Z2 = np.dot(W2, A1) + B2
    # (1,4)=(1,3)(3,4) + (1,1)
    A2 = 1 / (1 + np.exp(-Z2))
    # (1,4)
    # print('A2 is ', A2)

    dz2 = A2 - y
    # (1,4) = ( 1,4) - ( 1,4)
    # print('dz2 is ', dz2)
    dw2 = (1/4) * np.dot(dz2, A1.T)
    # (1,3) = (1,4)(4,3)
    # print('dw2 is ', dw2)
    db2 = (1/4) * np.sum(dz2, axis=1, keepdims=True)
    # (1,1) = (1,4) axis 1 rk2

    temp = np.dot(W2.T, dz2)
    dz1 = np.multiply(temp, A1_prime)
    # (3,4) = ((3,1).(1,4)) * (3,4)
    # print('dz1 is ', dz1)
    dw1 = (1/4) * np.dot(dz1, x)
    # (3,2) = ( 3,4).(4,2)
    # print('dw1 is ', dw1)
    db1 = (1/4) * np.sum(dz1, axis=1, keepdims=True)
    # (3,1) = (3,4) axis 1 rk2
    # print((np.sum(dw1, axis=1)))
    W1 = W1 - alpha * (np.sum(dw1, axis=1)).reshape(3, 1)
    W2 = W2 - alpha * dw2
    B1 = B1 - alpha * db1
    B2 = B2 - alpha * db2

while True:
    try:
        x1 = int(input('enter 1st bit'))
        x2 = int(input('enter 2nd bit'))
        x = np.array([[x1], [x2]])

        Z1 = (np.dot(W1, x) + B1)
        # print('z1 is ',Z1)
        A1 = 1/(1 + np.exp(Z1))

        # print('Ai is ',A1)

        Z2 = (np.dot(W2, A1) + B2)
        A2 = (1 / (1 + np.exp(Z2)))
        # (1,4m)
        print(A2)
        if A2 > 0.60:
            print("1")
        else:
            print("0")
        z = input("x for exit :: enter to continue")
        if z == 'x':
            break
    except ValueError:
        print("invalid input")
