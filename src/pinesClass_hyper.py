import os
import sys
import json
import argparse
import numpy as np
import joblib as jl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import accuracy_score, classification_report, \
                            confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV

# Exploratory Data Analysis (EDA)
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Random Forest (RF)
from sklearn.ensemble import RandomForestClassifier

# Logistic Regression (LogR)
from sklearn.linear_model import LogisticRegression

# Gaussian Naive Bayes (GNB)
from sklearn.naive_bayes import GaussianNB

# SVM Classifier
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

PINE_NAME = [
  'Roads or Barefields',          # 0
  'Alfalfa',                      # 1
  'Corn-notill',                  # 2
  'Corn-mintill',                 # 3
  'Corn',                         # 4
  'Grass-pasture',                # 5
  'Grass-trees',                  # 6
  'Grass-pasture-mowed',          # 7
  'Hay-windrowed',                # 8
  'Oats',                         # 9
  'Soybean-notill',               # 10
  'Soybean-mintill',              # 11
  'Soybean-clean',                # 12
  'Wheat','Woods',                # 13
  'Buildings Grass Trees Drives', # 14
  'Stone Steel Towers'            # 15
]
def plt_attr(ylabel: str = None, xlabel: str = None,
             yscale: str = None, xscale: str = None):
  if xlabel is not None: plt.xlabel(xlabel)
  if xscale is not None: plt.xscale(xscale)
  if yscale is not None: plt.yscale(yscale)
  if ylabel is not None: plt.ylabel(ylabel)
  return

def clean(path = None, exts = [".pkl", ".json"]):
  if path is None:
    path = DATA_PATH
  for file in os.listdir(path):
    if any([file.endswith(ext) for ext in exts]):
      os.remove(os.path.join(path, file))

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
parser.add_argument("-T", "--tree", type=str, default=None, required=False,
                    help="Path to Train the Model by determining whether "
                         "the input is a Tree or not and then the category it "
                         "belongs to. Default: None")
parser.add_argument("-r", "--review", action='store_true', default=False,
                    required=False, help="Inverse output up to PCA.")
parser.add_argument("--pca", type=int, required=False, default=42,
                    choices=range(0, 201), metavar="[0-200]",
                    help="Enable Principal Component Analysis (PCA) with n "
                         "components. If the value is set to zero, then LDA "
                         "will not be executed. Default: 42")
parser.add_argument("--lda", type=int, required=False, default=0,
                    choices=range(0, 201), metavar="[0-200]",
                    help="Enable Linear Discrimination Analysis (LDA) with n "
                         "components to be excecute after PCA. If the value "
                         "is set to zero, then LDA will not be executed. "
                         "Default: 0")
parser.add_argument("--RF", required=False, default=0, type=int,
                    help="Enable model Random Forest (RF) after Data "
                         "Preprocessing.")
parser.add_argument("--LogR", required=False, default=0, type=int,
                    help="Enable Logistic Regression (LogR) after Data "
                         "Preprocessing.")
parser.add_argument("--SVC", required=False, default=False,
                    action="store_true",
                    help="Flag which enables to run SVC after Data "
                         "Preprocessing.")
parser.add_argument("--GNB", required=False, default=False,
                    action="store_true",
                    help="Flag which enables to run Gaussian Naive Bayes "
                         "after Data Preprocessing.")

def plt_im(y, path, name, cmap="nipy_spectral"):
  plt.figure(figsize=(10, 8))
  plt.imshow(y, cmap=cmap)
  plt.colorbar()
  plt.axis("off")
  plt.title(name)
  plt.savefig(os.path.join(path, name + ".png"))
  plt.close()  # Close the figure to release resources
  return

def sns_im(y, path: str, name: str, cmap: str = "coolwarm"):
  plt.figure(figsize=(1000/myDPI, 800/myDPI), dpi=myDPI)
  sns.heatmap(y, cmap=cmap, annot=False)
  plt.title(name)
  plt.savefig(os.path.join(path, name + ".png"), dpi=myDPI*10)
  plt.close()  # Close the figure to release resources
  return

def plt_pl(y, path: str, name:str, color: str = "blue", x = None,
           ylabel: str = None, xlabel: str = None, yscale: str = None,
           xscale: str = None):
  plt.figure(figsize=(12, 6))
  if x is None:
    plt.plot(y, marker='o', linewidth=2, color=color)
  else:
    plt.plot(x, y, marker='o', linewidth=2, color=color)
  plt_attr(ylabel, xlabel, yscale, xscale)
  plt.title(name)
  plt.savefig(os.path.join(path, name + ".png"))
  plt.close()  # Close the figure to release resources
  return

def plt_sc(x, y, path: str, name: str, ylabel: str = None, xlabel: str = None,
           yscale: str = None, xscale: str = None, c = None, z = None,
           zlabel: str = None):
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
  plt.close()  # Close the figure to release resources
  return

def main(args):
  myDataSet = os.path.join(DATA_PATH, TARGET_STR + DATASET_STR + ".pkl")
  # Exploratory Data Analysis (EDA)
  SECTION = "EDA"
  if args.force or not os.path.isfile(myDataSet):
    clean()
    from scipy.io import loadmat
    # Load Dataset and Truth
    X = loadmat(DATASET_PATH)[DATASET_NAME]
    X_SHAPE = X.shape
    FEATURES = [f"Band_{i:03}" for i in range(1, X_SHAPE[2] + 1)]
    X = pd.DataFrame(X.reshape(-1, X_SHAPE[2]), columns=FEATURES)
    X[TARGET_STR] = loadmat(TARGET_PATH)[TARGET_NAME].flatten()
    X[FEATURES + [TARGET_STR]].to_pickle(myDataSet)
    with open(os.path.join(DATA_PATH, f"{TARGET_STR}_Datashape.json"),
              'w') as fp:
      json.dump({x: y for x, y in zip(['X', 'Y', 'Z'], X_SHAPE)}, fp)

    ## Plot truth
    plt_im(X[TARGET_STR].to_numpy().reshape(X_SHAPE[:2]),
          os.path.join(IMG_PATH, SECTION), TARGET_STR)
  else:
    X = pd.read_pickle(myDataSet)
    with open(os.path.join(DATA_PATH, f"{TARGET_STR}_Datashape.json"),
              'r') as fp:
      myDict = json.load(fp)
      X_SHAPE = [myDict[x] for x in ['X', 'Y', 'Z']]
    FEATURES = X.columns.values[:-1]

  if args.tree is None:
    X = X[X[TARGET_STR] != 0]
    X = X[X[TARGET_STR] != 1]
    X = X[X[TARGET_STR] != 7]
    X = X[X[TARGET_STR] != 9]

  scaler = StandardScaler().set_output(transform="pandas")
  if args.test != 0.0 and args.test != 1.0:
    # TRAIN + TEST
    # Standarize Dataset
    X_train, X_test, y_train, y_test = train_test_split(
      X[FEATURES], X[TARGET_STR], test_size=args.test, random_state=1,
      stratify=X[TARGET_STR])
    pd.concat([X_test, y_test]).to_csv(
      os.path.join(DATA_PATH, f"{TARGET_STR}_DataTest.csv"))
    scaler.fit(X_train)
    jl.dump(scaler, os.path.join(DATA_PATH, f"{TARGET_STR}_scaler.gz"))
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
  elif args.test == 1.0:
    # TEST
    X_train = None
    y_train = None
    myData = pd.read_csv(os.path.join(DATA_PATH, f"{TARGET_STR}_DataTest.csv"))
    scaler = jl.load(os.path.join(DATA_PATH, f"{TARGET_STR}_scaler.gz"))
    X_test = scaler.transform(myData[FEATURES])
    y_test = myData[TARGET_STR]
  else:
    # TRAIN
    scaler.fit(X[FEATURES])
    X[FEATURES] = pd.DataFrame(scaler.transform(X[FEATURES]), columns=FEATURES)
    X.to_pickle(myDataSet)
    X_train = X[FEATURES]
    y_train = X[TARGET_STR]
    X_test = None
    y_test = None

  CORR = np.abs(X.corr())
  sns_im(CORR, os.path.join(IMG_PATH, SECTION), "Correlation_Mtx")
  FEATURES_RELEVANT = CORR[CORR[TARGET_STR] > 0.2]

  name = TARGET_STR
  if args.pca != 0:
    # Principal Component Analysis (PCA)
    pinesPCA = PCA(n_components=args.pca, svd_solver="full")
    name += f"_PCA_{args.pca:03}"
    myDataSet = os.path.join(DATA_PATH, DATASET_STR + name + ".pkl")
    if args.force or not os.path.isfile(myDataSet):
      pinesPCA.set_output(transform="pandas")
      X_train = pinesPCA.fit_transform(X_train)
      X_test = pinesPCA.transform(X_test)
      jl.dump(pinesPCA, os.path.join(DATA_PATH, name + ".gz"))
      pd.concat([X_train, y_train], axis=1).to_pickle(myDataSet)
      PC1, PC2, PC3 = X_train[
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
      myData = pd.read_pickle(myDataSet)
      X_train = myData[[C for C in myData.columns if C != TARGET_STR]]
      y_train = myData[TARGET_STR]
      if X_test is not None:
        pinesPCA = jl.load(os.path.join(DATA_PATH, name + ".gz"))
        X_test = pinesPCA.transform(X_test)

  if args.lda:
    # Linear Discrimination Analysis (LDA)
    pinesLDA = None
    if name == "":
      name += "LDA"
    else:
      name += "_LDA"
    name += f"_{args.lda:03}"
    myDataSet = os.path.join(DATA_PATH, DATASET_STR + name + ".pkl")
    if args.force or not os.path.isfile(myDataSet):
      pinesLDA = LinearDiscriminantAnalysis(
                   solver="svd",
                   n_components=args.lda).set_output(transform="pandas")
      X_train = pinesLDA.fit_transform(X_train, y_train)
      X_test = pinesLDA.transform(X_test)
      jl.dump(pinesLDA, os.path.join(DATA_PATH, name + ".gz"))
      pd.concat([X_train, y_train], axis=1).to_pickle(myDataSet)
      LC1, LC2, LC3 = X_train[
                        pinesLDA.get_feature_names_out()[:3]].to_numpy().T
      plt_pl(pinesLDA.explained_variance_ratio_,
             os.path.join(IMG_PATH, SECTION), name + "_VarExp", yscale="log",
             xlabel="Linear Components", ylabel="Variance Explained")
      plt_sc(LC1, LC2, os.path.join(IMG_PATH, SECTION),
             name + "_" + TARGET_STR, c=y_train, z=LC3,
             xlabel="Linear Component 1",
             ylabel="Linear Component 2",
             zlabel="Linear Component 3")
    else:
      myData = pd.read_pickle(myDataSet)
      X_train = myData[[C for C in myData.columns if C != TARGET_STR]]
      y_train = myData[TARGET_STR]
      if X_test is not None:
        pinesLDA = jl.load(os.path.join(DATA_PATH, name + ".gz"))
        X_test = pinesLDA.transform(X_test)

  # WARNING: If the following assertion fails, then the Data has not been
  #          preprocessed
  assert name != TARGET_STR
  y_pred = dict()
  # CLASSIFIERS
  if args.RF:
    # Random Forest
    model_name = name + "_RF"
    pinesRF = RandomForestClassifier()
    param_grid = {
      #'n_estimators': range(100, 500, args.RF),
      'n_estimators': range(20,500,args.RF),
      'max_features': ['auto', 'sqrt', 'log2'],
      'max_depth' : range(15, 50, 5),
      'criterion' : ['gini', 'entropy', 'log_loss'],
      'bootstrap' : [True, False]
    }
    score_options = ['accuracy','f1','precision','recall']
    for score in score_options:
      print(score)
      CVpinesRF = GridSearchCV(estimator=pinesRF, scoring=score, param_grid=param_grid, cv=5,
                            n_jobs=-1)
      CVpinesRF.fit(X_train, y_train)
      print(CVpinesRF.best_params_)
      jl.dump(pinesRF, os.path.join(DATA_PATH, model_name + ".gz"))
      if X_test is not None:
        # We now test our model
        y_pred[model_name] = pinesRF.predict(X_test)

  if args.SVC:
    #
    model_name = name + "_SVC"
    # Initialize SVM classifier with a linear kernel
    pinesSVC = SVC(kernel='linear', C=1.0, random_state=42)
    # pinesSVC = SVC(gamma='auto')
    pinesSVC.fit(X_train, y_train)
    jl.dump(pinesSVC, os.path.join(DATA_PATH, model_name + ".gz"))
    if X_test is not None:
      # We now test our support vector model
      y_pred[model_name] = pinesSVC.predict(X_test)

  if args.LogR:
    # Logistic Regression
    model_name = name + "_LogR"
    pinesLogR = LogisticRegression(penalty=None, fit_intercept=True,
                                   max_iter=args.LogR, tol=1E-5)
    pinesLogR.fit(X_train, y_train)
    jl.dump(pinesLogR, os.path.join(DATA_PATH, model_name + ".gz"))
    if X_test is not None:
      # We now test our model
      y_pred[model_name] = pinesLogR.predict(X_test)

  if args.GNB:
    # Gaussian Naive Bayes
    model_name = name + "_GNB"
    pinesGNB = GaussianNB()
    pinesGNB.fit(X_train, y_train)
    jl.dump(pinesGNB, os.path.join(DATA_PATH, model_name + ".gz"))
    if X_test is not None:
      # We now test our support vector model
      y_pred[model_name] = pinesGNB.predict(X_test)
    pass

  # Test Reports
  if y_pred and y_test is not None:
    for model, pred in y_pred.items():
      print(model)
      # Evaluate the performance
      accuracy = accuracy_score(pred, y_test)
      print(f'Accuracy: {accuracy * 100:.2f}%')
      print(classification_report(pred, y_test, zero_division=1))

  return

if __name__ == "__main__": main(parser.parse_args())