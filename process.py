import pandas as pd
import numpy as np
import part2_house_value_regression as lib
from sklearn.model_selection import train_test_split


output_label = "median_house_value"
data = pd.read_csv("housing.csv")
x = data.loc[:, data.columns != output_label]
y = data.loc[:, [output_label]]
regressor = lib.Regressor(x)
xv = x.values
yv = y.values
x_train, x_test, y_train, y_test = train_test_split(
    xv, yv, test_size=0.2, random_state=0)
x_train = pd.DataFrame(data=x_train, columns=[x.columns])
x_test = pd.DataFrame(data=x_test, columns=[x.columns])
y_train = pd.DataFrame(data=y_train, columns=[y.columns])
y_test = pd.DataFrame(data=y_test, columns=[y.columns])

regressor = lib.Regressor(x_train, nb_epoch=10, batch_size=30, learning_rate=0.01, loss_fun="mse", neurons=[32,32, 1],
                          activations=["relu", "relu", "identity"])
regressor.fit(x_train, y_train)

#Error
error = regressor.score(x_test, y_test)

print("\nRegressor error: {}\n".format(error))


#results = pd.DataFrame(data=results, columns=['ground truth', 'predictions'])
#print(results)
#results.to_excel(r'\Results.xlsx', index=False)