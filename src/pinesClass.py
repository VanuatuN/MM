import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import train_test_split

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

DATA_PATH = "../data"
IMG_PATH = "../img"

TARGET_STR = "Target"
TARGET_PATH = os.path.join(DATA_PATH, "Indian_pines_gt.mat")
TARGET_NAME = "indian_pines_gt"
DATASET_STR = "Dataset"
DATASET_PATH = os.path.join(DATA_PATH, "Indian_pines_corrected.mat")
DATASET_NAME = "indian_pines_corrected"

def restricted_float(x):
  try:
    x = float(x)
  except ValueError:
    raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

  if x < 0.0 or x > 1.0:
    raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
  return x

parser = argparse.ArgumentParser(prog="Classify the pines based on Bands")
parser.add_argument("-f", "--force", action='store_true', default=False,
                    required=False,
                    help="Forces to compute everything, without considering"
                         "previuos computed results")
parser.add_argument("-t", "--test", type=restricted_float, default=0.3,
                    required=False,
                    help="")

# Initializing Classifiers
#clf1 = LogisticRegression(random_state=1, solver='lbfgs')
#clf2 = RandomForestClassifier(n_estimators=100,
#                              random_state=1)
#clf3 = GaussianNB()
#clf4 = SVC(gamma='auto')

def plt_im(y, path, name, cmap="nipy_spectral"):
  plt.figure(figsize=(10, 8))
  plt.imshow(y, cmap=cmap)
  plt.colorbar()
  plt.axis("off")
  plt.title(name)
  plt.savefig(os.path.join(path, name + ".png"))
  return

def sns_im(y, path, name, cmap="coolwarm"):
  plt.figure(figsize=(15, 10))
  sns.heatmap(y, cmap=cmap, annot=False)
  plt.title(name)
  plt.savefig(os.path.join(path, name + ".png"))
  return

def plt_pl():
  return

def main(args):
  myDataSet = os.path.join(DATA_PATH, DATASET_STR + ".pkl")
  # Exploratory Data Analysis (EDA)
  SECTION = "EDA"
  if args.force or not os.path.isfile(myDataSet):
    from scipy.io import loadmat
    # Load Dataset and Truth
    X = loadmat(DATASET_PATH)[DATASET_NAME]
    X_SHAPE = X.shape
    FEATURES = [f"Band_{i:03}" for i in range(1, X_SHAPE[2] + 1)]
    X = pd.DataFrame(X.reshape(-1, X_SHAPE[2]), columns=FEATURES)
    X[TARGET_STR] = loadmat(TARGET_PATH)[TARGET_NAME].flatten()
    X[FEATURES + [TARGET_STR]].to_pickle(myDataSet)
    with open(os.path.join(DATA_PATH, "Datashape.json"), 'w') as fp:
      json.dump({x: y for x, y in zip(['X', 'Y', 'Z'], X_SHAPE)}, fp)

    ## Plot truth
    plt_im(X[TARGET_STR].to_numpy().reshape(X_SHAPE[:2]),
          os.path.join(IMG_PATH, SECTION), TARGET_STR)
  else:
    X = pd.read_pickle(myDataSet)
    FEATURES = X.columns.values[:-1]

  scaler = StandardScaler()
  if args.test != 0.0 and args.test != 1.0:
    # TRAIN + TEST
    # Standarize Dataset
    X_train, X_test, y_train, y_test = train_test_split(
      X[FEATURES], X[TARGET_STR], test_size=args.test, random_state=1,
      stratify=X[TARGET_STR])
    scaler.fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), columns=FEATURES)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=FEATURES)
  elif args.test == 1.0:
    # TEST
    pass
  else:
    # TRAIN
    scaler.fit(X[FEATURES])
    X[FEATURES] = pd.DataFrame(scaler.transform(X[FEATURES]), columns=FEATURES)
    X.to_pickle(myDataSet)
    X_train = X[FEATURES]
    y_train = X[TARGET_STR]

  sns_im(X_train.corr(), os.path.join(IMG_PATH, SECTION), "Correlation_Mtx")

  # Principal Component Analysis (PCA)
  pinesPCA = PCA(svd_solver="full")
  pinesPCA.set_output(transform="pandas")
  XpinesPCA = pinesPCA.fit_transform(X_train)
  print(XpinesPCA)

  # Linear Discrimination Analysis (LDA)
  pinesLDA = LinearDiscriminantAnalysis().set_output(transform="pandas")
  XpinesLDA = pinesLDA.fit_transform(X_train, y_train)
  print(XpinesLDA)
  return

if __name__ == "__main__": main(parser.parse_args())