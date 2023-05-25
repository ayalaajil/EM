import pandas as pd 
import numpy as np
from algo_EM import emsem_function

dt = pd.read_csv("data_fraude.csv")

X1 = np.array(pd.DataFrame(dt[["newbalanceDest","oldbalanceDest","amount"]]))
X2 = np.array(pd.DataFrame(dt[["newbalanceOrig","oldbalanceOrg"]]))
Y = np.array(pd.DataFrame(dt["isFraud"]))

# Define the dimensions of the covariate matrices
n = Y.shape[0]  # Number of units/observations
rT = 2  # Dimension of covariate matrix T
r1 = 2  # Dimension of covariate matrix T1
r2 = 2  # Dimension of covariate matrix T2

# Simulate covariate matrices T, T1, and T2
T = np.random.normal(0, 1, size=(n, rT))  # Simulate matrix T
T1 = np.random.normal(0, 1, size=(n, r1))  # Simulate matrix T1
T2 = np.random.normal(0, 1, size=(n, r2))  # Simulate matrix T2


print(emsem_function(Y,X1,X2,T,T1,T2,epsilon=10**(-3),nb_it=100))

