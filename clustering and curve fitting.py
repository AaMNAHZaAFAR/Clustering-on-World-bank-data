

# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 11:39:59 2022

@author: RajaI
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
import scipy.optimize as opt
import numpy as np
import itertools as iter
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

# iPlot three clusters
plt.scatter(X_pca[y_kmeans == 0, 0], X_pca[y_kmeans == 0, 1], c='magenta',
            label='Cluster 1')
plt.scatter(X_pca[y_kmeans == 1, 0], X_pca[y_kmeans == 1, 1], c='blue',
            label='Cluster 2')
plt.scatter(X_pca[y_kmeans == 2, 0], X_pca[y_kmeans == 2, 1], c='green',
            label='Cluster 3')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='black', label='Centroids', s=100, marker="*")

plt.title("K means Clustering")
plt.legend()
plt.xlabel('CO2 Emission data 1990', fontsize=12)
plt.ylabel('CO2 Emission data 2019', fontsize=12)
plt.grid(True)
plt.show()

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


def logistics(t, scale, growth, t0):
    """ Computes logistics function with scale, growth raat
    and time of the turning point as free parameters
    """

    f = scale / (1.0 + np.exp(-growth * (t - t0)))

    return f


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
param, covar = opt.curve_fit(logistics, DATA["Years"],
                             DATA["Emission"])
print("Fit parameter", param)
DATA["log"] = logistics(DATA["Years"], *param)

plt.figure()
plt.plot(DATA["Years"], DATA["Emission"], label="data")
plt.plot(DATA["Years"], DATA["log"], label="fit")

plt.legend()
plt.title("First fit attempt")
plt.xlabel("year")
plt.ylabel("population")
plt.show()
print()


# estimated turning year: 2012
# Co2 emission in 2012: about 7.04
param = [7.04, 0.02, 2012]
DATA["log"] = logistics(DATA["Years"], *param)

plt.figure()
plt.plot(DATA["Years"], DATA["Emission"], label="data")
plt.plot(DATA["Years"], DATA["log"], label="fit")

plt.legend()
plt.xlabel("year")
plt.ylabel("population")
plt.title("Improved start value")
plt.show()


param, covar = opt.curve_fit(logistics, DATA["Years"],  DATA["Emission"],
                             p0=[7.04, 0.02, 2012])
print("Fit parameter", param)

DATA["log"] = logistics(DATA["Years"], *param)

plt.figure()
plt.plot(DATA["Years"], DATA["Emission"], label="data")
plt.plot(DATA["Years"], DATA["log"], label="fit")

plt.legend()
plt.title("Final logistics function")
plt.xlabel("year")
plt.xlabel("year")
plt.ylabel("population")
plt.show()


def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.

    """

    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower

    uplow = []   # list to hold upper and lower limits for parameters
    for p, s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))

    pmix = list(iter.product(*uplow))

    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)

    return lower, upper


# Use err_ranges function to estimate lower and upper limits of the confidence range
param, covar = opt.curve_fit(logistics, DATA["Years"],  DATA["Emission"])
x_pred = np.linspace(2021, 2040, 20)

# calculate the standard deviation for each parameter
sigma = np.sqrt(np.diag(covar))
y_pred, y_pred_err = err_ranges(x_pred, logistics, param, sigma)
plt.figure()
plt.plot(DATA["Years"], DATA["Emission"], label="data")
plt.plot(DATA["Years"], DATA["log"], label="fit")
plt.fill_between(x_pred, y_pred, y_pred_err, color='pink', label='confidence interval')
plt.legend()
plt.xlabel('Year')
plt.ylabel('cereal_yield')
plt.show()