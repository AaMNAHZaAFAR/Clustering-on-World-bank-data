

# -*- coding: utf-8 -*-
"""
github link:https://github.com/AaMNAHZaAFAR/Clustering-on-World-bank-data

@author: RajaI
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.optimize import curve_fit
import numpy as np
from uncertainties import ufloat
from sklearn.preprocessing import MinMaxScaler
from warnings import filterwarnings
filterwarnings('ignore')


def Read_File(DATA1):
    '''
    Function to print original and transposed data
    DATA1 : Nmae of Dataset (World Bank)
   '''
# File in original Worldbank format
    originaldata = DATA1
# Doing the tranpose
    transposeddata = DATA1.set_index('Country Name').T

    return originaldata, transposeddata


DATA1 = pd.read_csv('C:\\Users\\RajaI\\Desktop\\amna\\climate change\\C02emission.csv')
print(Read_File(DATA1))


DATA = DATA1[["Country Name", "Indicator Code", "1990", "2019"]]
X = DATA.iloc[:, -2:]
X = X.dropna()


# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
X_pca = scaler.fit_transform(X.iloc[:, -2:])

# Visualising original and MinMax scaled data
fig, axes = plt.subplots(1, 2)
axes[0].scatter(X.iloc[:, 0], X.iloc[:, 1], c='red')
axes[0].set_title("Original data")
axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c='green')
axes[1].set_title("MinMax scaled data")
plt.show()
css = []

# Finding inertia on various k values to identify number of clusters
for i in range(1, 8):
    kmeans = KMeans(n_clusters=i, init='k-means++',
                    max_iter=100, n_init=10, random_state=0).fit(X_pca)
    css.append(kmeans.inertia_)

plt.plot(range(1, 8), css, 'bx-')
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('CSS')
plt.show()


# Applying Kmeans classifier
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=100, n_init=10,
                random_state=0)
y_kmeans = kmeans.fit_predict(X_pca)

labels = kmeans.labels_
# Adding columns to indicate cluster number of each record of dataset
X["cluster"] = kmeans.labels_
print(X.head(50))

# Calculate Silhoutte Score
score = metrics.silhouette_score(X_pca, labels, metric='euclidean')
# Print the score
print('Silhouetter Score: %.3f' % score)

# Plot three clusters
plt.scatter(X_pca[y_kmeans == 0, 0], X_pca[y_kmeans == 0, 1], c='magenta',
            label='Cluster 1')
plt.scatter(X_pca[y_kmeans == 1, 0], X_pca[y_kmeans == 1, 1], c='blue',
            label='Cluster 2')
plt.scatter(X_pca[y_kmeans == 2, 0], X_pca[y_kmeans == 2, 1], c='green',
            label='Cluster 3')
# Plot three centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='black', label='Centroids', s=100, marker="*")
plt.title("K means Clustering")
plt.legend()
plt.xlabel('CO2 Emission data 1990', fontsize=12)
plt.ylabel('CO2 Emission data 2019', fontsize=12)
plt.grid(True)
plt.show()


# Setting iteartions for better results
n_iter = 9
fig, ax = plt.subplots(3, 3, figsize=(16, 16))
ax = np.ravel(ax)
centers = []
for i in range(n_iter):
    # Run local implementation of kmeans
    kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=100, n_init=10,
                    random_state=0)
    y_kmeans = kmeans.fit_predict(X_pca)

    ax[i].scatter(X_pca[y_kmeans == 0, 0], X_pca[y_kmeans == 0, 1],
                  c='magenta', label='Cluster 1')
    ax[i].scatter(X_pca[y_kmeans == 1, 0], X_pca[y_kmeans == 1, 1], c='blue',
                  label='Cluster 2')
    ax[i].scatter(X_pca[y_kmeans == 2, 0], X_pca[y_kmeans == 2, 1], c='green',
                  label='Cluster 3')

    ax[i].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                  c='black', label='centroids', s=200, marker="*")
    ax[i].set_xlim([-2, 2])
    ax[i].set_ylim([-2, 2])
    ax[i].legend(loc='lower right')
    ax[i].set_aspect('equal')
plt.show()


# Curve fitting
def logistics(t, scale, growth, t0):
    """ Computes logistics function with scale, growth raat
    and time of the turning point as free parameters
    """

    f = scale / (1.0 + np.exp(-growth * (t - t0)))

    return f


# Data Preprocessing
DATA = DATA1[["Country Name", "Indicator Code", '1996', '1997', '1998', '1999',
             '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007',
              '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015',
              '2016', '2017', '2018', '2019']]
DATA = DATA[(DATA['Country Name'] == 'China')]
DATA = pd.melt(DATA, id_vars="Country Name",
               value_vars=['1996', '1997', '1998', '1999', '2000', '2001',
                           '2002', '2003', '2004', '2005', '2006', '2007',
                           '2008', '2009', '2010', '2011', '2012', '2013',
                           '2014', '2015', '2016', '2017', '2018', '2019'
                           ], var_name="Years",
               value_name="Emission")
print(DATA)

DATA["Years"] = DATA["Years"].astype(float)

# Fitting the logistic function
parameter, covar = curve_fit(logistics, DATA["Years"], DATA["Emission"])
print("Fit parameter at first attempt", parameter)
DATA["logistic"] = logistics(DATA["Years"], *parameter)
plt.figure()
plt.plot(DATA["Years"], DATA["Emission"], label="Original data")
plt.plot(DATA["Years"], DATA["logistic"], label="fit")
plt.title("First fit attempt")
plt.xlabel("year")
plt.ylabel("CO2 Emission data of China")
plt.legend()
plt.show()
print()


# Set estimated turning year: 2012
# Co2 emission in 2012: about 7.04
parameter = [7.04, 0.02, 2012]
DATA["logistic"] = logistics(DATA["Years"], *parameter)
plt.figure()
plt.plot(DATA["Years"], DATA["Emission"], label="Original data")
plt.plot(DATA["Years"], DATA["logistic"], label="fit")
plt.xlabel("year")
plt.ylabel("CO2 Emission data of China")
plt.title("Improved Curve fitting")
plt.legend()
plt.show()

# Best fit parameter
parameter, covar = curve_fit(logistics, DATA["Years"],  DATA["Emission"],
                             p0=[7.04, 0.02, 2012])
print("Fit parameter at final attempt ", parameter)
sigma_values = np.sqrt(np.diagonal(covar))
x = ufloat(parameter[0], sigma_values[0])
y = ufloat(parameter[1], sigma_values[1])
# Estimating forecast from 1995 to 2025
y_pred = np.linspace(1995, 2025, 20)
# Estimating best fit parameter
text_res = "Best fit parameters:\na = {}\nb = {}".format(x, y)
print(text_res)
plt.figure()
plt.plot(DATA["Years"], DATA["Emission"], label="Original data")
plt.plot(y_pred, logistics(y_pred, *parameter), 'red', label="fit")
# Upper bound and Lower bound for confidence intervals
bound_upper = logistics(y_pred, *(parameter + sigma_values))
bound_lower = logistics(y_pred, *(parameter - sigma_values))
# plotting the confidence intervals
plt.fill_between(y_pred, bound_lower, bound_upper,
                 color='black', alpha=0.15, label="Confidence")
plt.legend()
plt.title("Final attempt of logistics function")
plt.xlabel("year")
plt.ylabel("CO2 Emission data of China")
plt.show()
