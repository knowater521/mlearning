import sklearn
from sklearn.linear_model import LinearRegression

X = [[4], [8], [5], [10], [12]]
y = [20, 50, 30, 70, 60]

model = LinearRegression()
model.fit(X, y)
print(model.coef_)
print(model.intercept_)