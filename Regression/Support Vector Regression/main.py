# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv("./Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

print("X", X)
print("y", y)
## need to change y to 2D array because the standard scalar expects it to be 2D array
y = y.reshape(len(y), 1)
print("y", y)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)
print("X", X)
print("y", y)

# Training the SVR Model on the whole dataset
from sklearn.svm import SVR

regressor = SVR(kernel="rbf")
regressor.fit(X, y)

# Predicting a new result
## we will be scaling the input for the prediction because the data on which the model is trained is also scaled
## then we will use inverse scaling to get the output back in the actual data
print(sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1, 1)))

# Visualising the SVR Results
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color="red")
plt.plot(
    sc_X.inverse_transform(X),
    sc_y.inverse_transform(regressor.predict(X).reshape(-1, 1)),
    color="blue",
)
plt.title("Truth or Bluff (SVR)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
# Visualising the SVR Results ( for higher resolution and smoother curve )
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color="red")
plt.plot(X_grid,
         sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(-1, 1)),
         color='blue'
         )
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
