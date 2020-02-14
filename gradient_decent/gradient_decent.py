#y = theta0 + theta1*x
X = [4, 8, 5, 10, 12]
y = [20, 50, 30, 70, 60]

theta0 = theta1 = 0

alpha = 0.00001

cnt = 0

error0 = error1 = 0

threshold = 0.0000001

while True:
    diff = [0, 0]
    m = len(X)
    #BGD
    for i in range(m):
        diff[0] += y[i]-(theta0+theta1*X[i])
        diff[1] += (y[i]-(theta0+theta1*X[i]))*X[i]
    theta0 = theta0+alpha*diff[0]/m
    theta1 = theta1+alpha*diff[1]/m
    #SGD
    """
    for i in range(m):
        diff[0] = y[i]-(theta0+theta1*X[i])
        diff[1] = (y[i]-(theta0+theta1*X[i]))*X[i]
        theta0 = theta0+alpha*diff[0]
        theta1 = theta1+alpha*diff[1]
    """
    #MSGD
    """
    batch = 5
    for i in range(0, m, batch):
        diff[0] += y[i]-(theta0+theta1*X[i])
        diff[1] += (y[i]-(theta0+theta1*X[i]))*X[i]
    theta0 = theta0+alpha*diff[0]/batch
    theta1 = theta1+alpha*diff[1]/batch
    
    """

    for i in range(m):
        error1+=(y[i]-(theta0+theta1*X[i]))**2
    error1/=m
    if abs(error1-error0)<threshold:
        break
    else:
        error0=error1
    cnt+=1
    print("decent %d, theta0 %f, theta1 %f" %(cnt, theta0, theta1))
print(theta0, theta1, cnt)

#decent 198255, theta0 1.206861, theta1 5.739106
#1.2068624206638512 5.739106302152591 198255