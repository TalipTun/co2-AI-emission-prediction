import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 

# Loading the dataset
path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"

#pandas library
df = pd.read_csv(path) 

# Describing the dataset
print(df.head()) 
print(df.describe()) 

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']] 

# Visualizing the data
viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']] 
viz.hist() 

#matplotlib library 
plt.show() 
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue') 
plt.xlabel("FUELCONSUMPTION_COMB") 
plt.ylabel("Emission") 
plt.show() 

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Cylinders")
plt.ylabel("Emission")
plt.show()

# Splitting data into training and test sets
#numpy library
msk = np.random.rand(len(df)) < 0.8 
train = cdf[msk] 
test = cdf[~msk]

# Training the model
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.title("Training data")
plt.show()

from sklearn import linear_model 
#scikit-learn library:
regr = linear_model.LinearRegression() 

train_x = np.asanyarray(train[['ENGINESIZE']]) 
train_y = np.asanyarray(train[['CO2EMISSIONS']])
#scikit-learn library:
regr.fit(train_x, train_y) 

# The coefficients
print('Coefficients: ', regr.coef_) 
print('Intercept: ', regr.intercept_) 

# Plotting the regression line
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue') 
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r') 
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show() 

# Testing the model
from sklearn.metrics import r2_score 

test_x = np.asanyarray(test[['ENGINESIZE']]) 
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x) 

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))

print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))

print("R2-score: %.2f" % r2_score(test_y, test_y_))

# Example input
def predict_co2(engine_size):
    engine_size_array = np.array([[engine_size]])  
    predicted_co2 = regr.predict(engine_size_array) 
    return predicted_co2[0][0]  


input_engine_size = float(input("Enter the engine size to predict CO2 emissions: "))
predicted_co2 = predict_co2(input_engine_size)
print(f"Predicted CO2 emissions for engine size {input_engine_size}: {predicted_co2:.2f}")
