import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.io import loadmat
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

# Exploratory Data Analysis (EDA)
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Random Forest (RF)
from sklearn.ensemble import RandomForestClassifier

# Logistic Regression (LogR)
from sklearn.linear_model import LogisticRegression

# 
from sklearn.naive_bayes import GaussianNB

# 
from sklearn.svm import SVC

TARGET_PATH = "../data/Indian_pines_gt.mat"
TARGET_NAME = "indian_pines_gt"
DATASET_PATH = "../data/Indian_pines_corrected.mat"
DATASET_NAME = "indian_pines_corrected"

# Initializing Classifiers
clf1 = LogisticRegression(random_state=1, solver='lbfgs')
clf2 = RandomForestClassifier(n_estimators=100,
                              random_state=1)
clf3 = GaussianNB()
clf4 = SVC(gamma='auto')

#from sklearn.preprocessing import StandardScaler

def plot_im(y, name):
  plt.figure(figsize=(10, 8))
  plt.imshow(y, cmap="nipy_spectral")
  plt.colorbar()
  plt.axis("off")
  plt.savefig(name + ".png")

def main():
  # Load Dataset and Truth
  X = loadmat(DATASET_PATH)[DATASET_NAME]
  y = loadmat(TARGET_PATH)[TARGET_NAME]
  # Plot truth
  
  return

if __name__ == "__main__": main()