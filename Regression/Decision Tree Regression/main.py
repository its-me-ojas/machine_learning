# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv("../Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

print("X", X)
print("y", y)
# Training the Decision Tree Regression Model on the whole dataset
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)
# Predicting a new result
print(regressor.predict([[6.5]]))

# Visualising the Decision Tree Regression Results (high resolution)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color="red")
plt.plot(X_grid,
         regressor.predict(X_grid),
         color='blue'
         )
plt.title('Truth or Bluff (Decision Tree)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
