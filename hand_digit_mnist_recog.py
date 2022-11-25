########################################################################################
#LIBRARIES##############################################################################
########################################################################################

#Libraries for Data handling & plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# SciKit Learn Libraries for ML 
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

#Libraries for Image resizing
from cv2 import resize



########################################################################################
#Exploratory Analysis###################################################################
########################################################################################
#INITIAL CONSIDERATIONS:
#   * The dataset is already preprocessed ergo it does not need any preliminar procedure

#WHAT WE WANT TO ACHIEVE?
# Begin with an exploratory analysis of the data. Can you spot useless variables by 
# looking at their summary statisitcs? Consider the class distribution: what percentage 
# of cases would be classified correctly if we simply predict the majority class?   

#Importing and displaying the MNIST Dataset
mnist = pd.read_csv("type_your_mnist_local_path", sep=',', header='infer')
#print(mnist.head())

#Size of MNIST (42000, 785)
print(mnist.shape)
#Generate descriptive statistics (mean, std, percentiles)
print(mnist.describe())

#TODO: print a normal distribution of the majority class




