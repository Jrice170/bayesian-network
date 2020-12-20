
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC

df = pd.read_csv("Support_vector.csv")

X = df.iloc[:,0:2].values
y = df.iloc[:,2].values
r = df.shape

#plt.scatter(X[:,0],X[:,1],c=y,s=r,Cmap='winter')


model = SVC(kernel = 'poly',degree = 2)
model.fit(X,y)
print(model.support_vectors_)

print(model.intercept_)
print("\n")
print([[4,5]])
print(model.predict([[4,5]]))
print([[2,2]])
print(model.predict([[2,2]]))
print([[1,1]])
print(model.predict([[1,1]]))
print([[0,-0.5]])
print(model.predict([[0,-0.5]]))
