# MM
Group project - Machine learning (P.11)

Ken 
Edward 
Andrea 
Natalia

# Report

## Project strategy 

STEP 1 Exploratory Data Analysis (EDA)<br>

_1. Data Loading:_
The script loads the "Indian Pines" dataset and its ground truth labels.
The dataset includes information about different bands and their corresponding labels.<br>
_2. Data Preprocessing:_
The script performs data cleaning and preprocessing steps, including standardization using StandardScaler.
It checks if the data has been previously computed; if not, it processes the data and saves it for future use.<br>
3._Visualizations:_
Visualizations of the ground truth labels and correlation matrix are generated and saved in the IMG_PATH (/img) directory.<br>

STEP 2 Dimensionality Reduction (PCA and LDA)<br>

_4. Principal Component Analysis (PCA):_
If PCA is enabled (--pca option), the script applies PCA to reduce the dimensionality of the data.
Visualizations include the explained variance ratio plot and scatter plots of principal components.<br>
_5. Linear Discriminant Analysis (LDA):_
If LDA is enabled (--lda option), the script applies LDA for further dimensionality reduction.
Visualizations include the explained variance ratio plot and scatter plots of linear components.<br>

STEP 3: Model Training and Testing<br>

_6. Data Splitting:_
The dataset is split into training and testing sets based on the specified ratio (--test option).<br>
_7. Model Training:_
The script supports various classifiers such as Random Forest (--RF), Support Vector Classifier (--SVC), Logistic Regression (--LogR), and Gaussian Naive Bayes (--GNB).
Model training is performed using the training set, and hyperparameter tuning is conducted using GridSearchCV.<br>
_8. Model Testing:_
If a separate test set is specified, the trained models are applied to make predictions on the test set.<br>

STEP 4: Model Evaluation and Reporting<br>

_9. Model Evaluation:_
For each trained model, the script evaluates its performance using metrics like accuracy, classification report, confusion matrix,
precision, recall, and F1-score. These reports are displayed in the console. <br>

Generally, our script provides insights into the classification performance of different machine learning models on the "Indian Pines" dataset.<br>

## Backgorund
 
Hyperspectral data provide a lot of information for the remote discrimination of ground truth, however, since spectral dimensions are usually many, the possibility of information redundancy is presented. Data analysis and interpretation of hyperspectral images can also be a challenge. <br>

The goal of the group assignmnet was to explore machine learning tools to analyze hyperspectral images of Indian pine fields to classify land surfaces according to the groud truth proveded. <br>


The dataset consists of 200 satellite images of the same area, each corresponds to the one spectral band of the remote sensor. We expect different types of the land surface 
to have a different reflectivity among those 200 bands. We will make at attempt to classify land types according to their representation on images in different bands. <br>

We also have a "reference": the images that contains "target": classified patters of the surface, e.g. 'Corn-notill', 'Corn', etc. <br>
Assuming we trained our model on this dataset, e.g. managed to predic the type of the land surface on the satellte imagery this can further be applied
for the classification of the same 200 bands on the satellite imagery for the other areas. <br>

## Exploratory Data Analysis

_Important note_: All 0 values and values of the target that covered sparsely by the data were removed, or classified as NaNs. The sparsely covered
targets are: 0, 1, 7 and 9. In the end we analyse tagets: 2, 3, 4, 5, 6, 8, 10-16. 13 in total, each for one type of the land. 

![image](https://github.com/VanuatuN/MM/assets/23639320/bb881288-5bcd-4b7d-a19e-1010b8c00b24)
Figure 1: Binned distriburion of the image cells with different features (e.g. land types).

- **Principal Components Analysis** <br>

We first expore the data by plotting images for random bands. There are several patterns that can be observed from this simple procedure, this suggest 
some land types are clearly distinguishable in different satellite bands.

![image](https://github.com/VanuatuN/MM/assets/23639320/2e8cf3a1-93c0-4b81-a38f-46eaa62ec35b)

Figure 2. Example of the satellite images in different spectral bands. 

As a first step we apply a Pricipal Components decomposition to the 200 matrixes of the size 145x145 to see
whether PCs are (i) distiguashable between each other and (ii) how many PCs we need to describe most of the varibility
in the dataset. These anlysis allows to see the clusters in the data and quantify the measure of their
"separation" to make further descision for the methods of analysis. <br>

The PCs analysis shows that first 5 PCs expalin more than 92% of the total variability in the dataset.
While <br>

PC 1 explains 0.68 % <br>
PC 2 explains 0.19 % <br>

There is also a clear clustering of the data points in PCs space (Figure 2), suggesting that data clusters are
separated and can be further analysed succesfully with machine learning methods. 
![image](https://github.com/VanuatuN/MM/assets/23639320/bdf28b5c-0a2c-4b23-94bf-53932a34bddf)
Figure 3. First 3 PCAs plotted in a 3D space. <br>

The next step was to check whether the reconstucted images only applying first 10 PCs would
reflect the main features to be carptured by machine learning techniques. Figure 3 demonstrates
those reconstructed images and we conclude that images are well reflecting the land features
we want to classify. 
![image](https://github.com/VanuatuN/MM/assets/23639320/5ac60da7-c650-4483-96f6-c79b475088dc)
Figure 4. Reconstructed images (applying inverse transform with first 10 PCs) of for the different bands. <br>


Exploratory Data analysis of our choice focused on, first understanding the dataset probing the overall description of the dataset. Pixel sizes (data) contained in 200 bands of image were analyzed for the presence of redundancy of the data they all held.<br> 


This was achieved through the assesment of interband correlation. Of the first 15 bands, band1 had the weakest correlation with the remainig bands (bands2-band15), showing a very strong correlation between band2 to 15 with coefficients ranging between 0.7 to 0.9 in most combinations. <br>


The correlation coefficients of the bands with the class (specie) column was analyzed. The highest correlation coefficient was estimated to be ~ 0.23. Selected Bands with Correlation Coefficient >= 0.238 with the Class (Specie) Column were as follows:<br>

| Band ID | Correlation Coefficient with the Class Column |
|---------|----------------------------------------------|
| band147 | 0.245247                                     |
| band148 | 0.245009                                     |
| band149 | 0.242812                                     |
| band150 | 0.242855                                     |
| band151 | 0.238947                                     |
| band153 | 0.238003                                     |
| band155 | 0.239565                                     |
| band184 | 0.238006                                     |
| band185 | 0.241086                                     |
| band188 | 0.238426                                     |
| band190 | 0.239321                                     |
| band191 | 0.238504                                     |
| band192 | 0.239755                                     |
| band193 | 0.241024                                     |
| band194 | 0.242920                                     |
| band195 | 0.238310                                     |
| band196 | 0.240277                                     |


It was obvious that these bands were strongly correlated as well, hence any two of them, could most probably be used to train an algorithm to make predictions. <br>

A plot of the pixel distribution of the 'Class' column for band196  is presented below:<br>
![Alt text](band196_vs_class.png) <br>

Figure 5: Band 196 vs Class <br>

- **Linear Discriminant Analysis** <br>

Figure 6a and 6b show a simple Linear Discriminant Analysis (LDA) and a t-Distributed Stochastic Neighbor Embedding (t-SNE) was used to visualize the high-dimensional raw data in lower-dimensional spaces, typically 3D and 2D respectively. <br> 

![Alt text](lda_raw.png) <br>
Figure 6a.

![Alt text](tSNE_raw.png) <br>
Figure 6b.

Rememer we dropped the class '0', based on these preliminary data analysis on the raw dataset as well as other sparsely covered with data classes. <br> 
The modified dataset is then standardized, fitted, transformed and a binary classification is performed on it using the Random Forest classifier. We consider only the output of the model which predicts the presence of pine species, to help the next multivariate classification and improve the accuracy score of the prediction. <br>

- Linear Discriminant Analysis <br>
It is a technique to reduce the dimensionality and help classification, by finding the linear combinations of features that best separate the different classes in the dataset.
It is best employed before the application of a classificaton algorithm, by maximizing the distance between the means of different classes and minimizing the spread within each class, thus enhancing the discriminatory power of the features and the accuracy of the classification.


The script checks if the dataset has been preprocessed. If not, it loads the dataset, applies standardization, and saves the preprocessed dataset.
EDA (Exploratory Data Analysis):
The script performs exploratory data analysis, including plotting the ground truth (target) and correlation matrix.

The dataset is split into training and testing sets. Standardization (scaling) is applied using StandardScaler.

We apply Principal Component Analysis to reduce dimensionality. It plots the explained variance ratio and scatter plots of principal components.

If LDA is enabled, the script applies Linear Discriminant Analysis. It plots the explained variance ratio and scatter plots of linear components.

Random Forest, Logistic Regression, Support Vector Classifier (SVC), and Gaussian Naive Bayes classifiers are trained. Hyperparameter tuning is performed using GridSearchCV.

If testing is required, the trained models are used to predict the target values on the test set.

Accuracy scores and classification reports are printed for each enabled model.

Visualization:
Various visualizations are generated, such as correlation matrices, PCA/LDA variance explained plots, and scatter plots.
Execution Check:
An assertion check ensures that data preprocessing has been performed before model training.


## Classification report
### Random Forest
Run: python pinesClass.py -RF #

### Logistic Regression (LogR)


### Support Vector Classification (SVC)


### Gaussian Naive Bayes (Gaussian NB)



In supervised learning, a training data set consisting of input–output pairs is available, and a Machine Learning algorithm is trained with the goal of providing predictions of the desired output for unseen input values.
