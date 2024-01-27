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

myDPI = 96

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
                    help="Forces to compute everything, without considering "
                         "previuos computed results. Default: False")
parser.add_argument("-t", "--test", type=restricted_float, default=0.3,
                    required=False, metavar="[0, 1]",
                    help="Size ratio between Testing and Training Datasets. "
                         "If this ratio is set to zero, then the program will "
                         "Train in the entire Dataset. Default: 0.3")
parser.add_argument("--pca", type=int, required=False, default=42,
                    choices=range(0, 201), metavar="[0-200]",
                    help="Enable Principal Component Analysis (PCA) with n "
                         "components. Default: 42")

# Initializing Classifiers
#clf1 = LogisticRegression(random_state=1, solver='lbfgs')
#clf2 = RandomForestClassifier(n_estimators=100,
#                              random_state=1)
#clf3 = GaussianNB()
#clf4 = SVC(gamma='auto')

def plt_im(y, path, name, cmap = "nipy_spectral"):
  plt.figure(figsize=(10, 8))
  plt.imshow(y, cmap=cmap)
  plt.colorbar()
  plt.axis("off")
  plt.title(name)
  plt.savefig(os.path.join(path, name + ".png"))
  return

def sns_im(y, path:str, name:str, cmap:str = "coolwarm"):
  plt.figure(figsize=(1000/myDPI, 800/myDPI), dpi=myDPI)
  sns.heatmap(y, cmap=cmap, annot=False)
  plt.title(name)
  plt.savefig(os.path.join(path, name + ".png"), dpi=myDPI*10)
  return

def plt_attr(ylabel:str = None, xlabel:str = None,
             yscale: str = None, xscale: str = None):
  if xlabel is not None:
    plt.xlabel(xlabel)
  if xscale is not None:
    plt.xscale(xscale)
  if yscale is not None:
    plt.yscale(yscale)
  if ylabel is not None:
    plt.ylabel(ylabel)
  return

# TODO: Use kwargs
def plt_pl(y, path:str, name:str, color:str = "blue", x = None,
           ylabel:str = None, xlabel:str = None, yscale: str = None,
           xscale: str = None):
  plt.figure(figsize=(12, 6))
  if x is None:
    plt.plot(y, marker='o', linewidth=2, color=color)
  else:
    plt.plot(x, y, marker='o', linewidth=2, color=color)
  plt_attr(ylabel, xlabel, yscale, xscale)
  plt.title(name)
  plt.savefig(os.path.join(path, name + ".png"))
  return

# TODO: Use kwargs
def plt_sc(x, y, path:str, name:str, ylabel:str = None, xlabel:str = None,
           yscale: str = None, xscale: str = None, c = None, z = None,
           zlabel = None):
  if z is None:
    plt.scatter(x, y, c=c)
    plt_attr(ylabel, xlabel, yscale, xscale)
    plt.title(name)
  else:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, z, c=c)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
  plt.savefig(os.path.join(path, name + ".png"))
  return

def main(args):
  myDataSet = os.path.join(DATA_PATH, DATASET_STR + ".pkl")
  # Exploratory Data Analysis (EDA)
  SECTION = "EDA"
  if args.force or not os.path.isfile(myDataSet):
    # TODO: Remove all '.pkl' & '.json' files
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
    with open(os.path.join(DATA_PATH, "Datashape.json"), 'r') as fp:
      myDict = json.load(fp)
      X_SHAPE = [myDict[x] for x in ['X', 'Y', 'Z']]
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

  sns_im(X.corr(), os.path.join(IMG_PATH, SECTION), "Correlation_Mtx")
  FEATURES_RELEVANT = np.abs(X.corr()[TARGET_STR]) > 0.5
  if args.pca != 0:
    # Principal Component Analysis (PCA)
    name = f"PCA_{args.pca:03}"
    myDataSet = os.path.join(DATA_PATH, DATASET_STR + name + ".pkl")
    if args.force or not os.path.isfile(myDataSet):
      pinesPCA = PCA(n_components=args.pca, svd_solver="full")
      pinesPCA.set_output(transform="pandas")
      XpinesPCA = pinesPCA.fit_transform(X_train)
      XpinesPCA.to_pickle(myDataSet)
      PC1, PC2, PC3 = XpinesPCA[
                        pinesPCA.get_feature_names_out()[:3]].to_numpy().T
      plt_pl(pinesPCA.explained_variance_ratio_,
             os.path.join(IMG_PATH, SECTION), name + "_VarExp", yscale="log",
             xlabel="Principal Components", ylabel="Variance Explained")
      plt_sc(PC1, PC2, os.path.join(IMG_PATH, SECTION),
             name + "_" + TARGET_STR, c=y_train, z=PC3,
             xlabel="Principal Component 1",
             ylabel="Principal Component 2",
             zlabel="Principal Component 3")
    else:
      XpinesPCA = pd.read_pickle(myDataSet)

  else:
    pass

  # Linear Discrimination Analysis (LDA)
  pinesLDA = LinearDiscriminantAnalysis().set_output(transform="pandas")
  XpinesLDA = pinesLDA.fit_transform(X_train, y_train)
  print(XpinesLDA)
  return

if __name__ == "__main__": main(parser.parse_args())