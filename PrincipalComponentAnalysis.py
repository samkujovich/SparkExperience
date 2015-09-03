# **Principal Component Analysis Lab**

# *Part 1:* Work through the steps of PCA on a sample dataset
# *Visualization 1:* Two-dimensional Gaussians
# *Part 2:* Write a PCA function and evaluate PCA on sample datasets

labVersion = 'cs190_week5_v_1_2'



# ** Work through the steps of PCA on a sample dataset**


import matplotlib.pyplot as plt
import numpy as np

def preparePlot(xticks, yticks, figsize=(10.5, 6), hideLabels=False, gridColor='#999999',
                gridWidth=1.0):
    """Template for generating the plot layout."""
    plt.close()
    fig, ax = plt.subplots(figsize=figsize, facecolor='white', edgecolor='white')
    ax.axes.tick_params(labelcolor='#999999', labelsize='10')
    for axis, ticks in [(ax.get_xaxis(), xticks), (ax.get_yaxis(), yticks)]:
        axis.set_ticks_position('none')
        axis.set_ticks(ticks)
        axis.label.set_color('#999999')
        if hideLabels: axis.set_ticklabels([])
    plt.grid(color=gridColor, linewidth=gridWidth, linestyle='-')
    map(lambda position: ax.spines[position].set_visible(False), ['bottom', 'top', 'left', 'right'])
    return fig, ax

def create2DGaussian(mn, sigma, cov, n):
    """Randomly sample points from a two-dimensional Gaussian distribution"""
    np.random.seed(142)
    return np.random.multivariate_normal(np.array([mn, mn]), np.array([[sigma, cov], [cov, sigma]]), n)


dataRandom = create2DGaussian(mn=50, sigma=1, cov=0, n=100)

# generate layout and plot data
fig, ax = preparePlot(np.arange(46, 55, 2), np.arange(46, 55, 2))
ax.set_xlabel(r'Simulated $x_1$ values'), ax.set_ylabel(r'Simulated $x_2$ values')
ax.set_xlim(45, 54.5), ax.set_ylim(45, 54.5)
plt.scatter(dataRandom[:,0], dataRandom[:,1], s=14**2, c='#d6ebf2', edgecolors='#8cbfd0', alpha=0.75)
pass


dataCorrelated = create2DGaussian(mn=50, sigma=1, cov=.9, n=100)

# generate layout and plot data
fig, ax = preparePlot(np.arange(46, 55, 2), np.arange(46, 55, 2))
ax.set_xlabel(r'Simulated $x_1$ values'), ax.set_ylabel(r'Simulated $x_2$ values')
ax.set_xlim(45.5, 54.5), ax.set_ylim(45.5, 54.5)
plt.scatter(dataCorrelated[:,0], dataCorrelated[:,1], s=14**2, c='#d6ebf2',
            edgecolors='#8cbfd0', alpha=0.75)
pass


# ** Interpreting PCA **

correlatedData = sc.parallelize(dataCorrelated)

meanCorrelated = correlatedData.mean()
correlatedDataZeroMean = correlatedData.map(lambda x: x-meanCorrelated)

print meanCorrelated
print correlatedData.take(1)
print correlatedDataZeroMean.take(1)

# TEST Interpreting PCA
from test_helper import Test
Test.assertTrue(np.allclose(meanCorrelated, [49.95739037, 49.97180477]),
                'incorrect value for meanCorrelated')
Test.assertTrue(np.allclose(correlatedDataZeroMean.take(1)[0], [-0.28561917, 0.10351492]),
                'incorrect value for correlatedDataZeroMean')


# **Sample covariance matrix**

correlatedCov = correlatedDataZeroMean.map(lambda x: np.outer(x,x)).mean()
print correlatedCov

# TEST Sample covariance matrix
covResult = [[ 0.99558386,  0.90148989], [0.90148989, 1.08607497]]
Test.assertTrue(np.allclose(covResult, correlatedCov), 'incorrect value for correlatedCov')


# ** Covariance Function **

def estimateCovariance(data):
    """Compute the covariance matrix for a given rdd.

    Note:
        The multi-dimensional covariance array should be calculated using outer products.  Don't
        forget to normalize the data by first subtracting the mean.

    Args:
        data (RDD of np.ndarray):  An `RDD` consisting of NumPy arrays.

    Returns:
        np.ndarray: A multi-dimensional array where the number of rows and columns both equal the
            length of the arrays in the input `RDD`.
    """
    meandata = data.mean()
    dataZeroMean = data.map(lambda x: x-meandata)
    return dataZeroMean.map(lambda x: np.outer(x,x)).mean()
    
correlatedCovAuto= estimateCovariance(correlatedData)
print correlatedCovAuto

# TEST Covariance function
correctCov = [[ 0.99558386,  0.90148989], [0.90148989, 1.08607497]]
Test.assertTrue(np.allclose(correctCov, correlatedCovAuto),
                'incorrect value for correlatedCovAuto')


# **Eigendecomposition **

from numpy.linalg import eigh

# Calculate the eigenvalues and eigenvectors from correlatedCovAuto
eigVals, eigVecs = eigh(correlatedCovAuto)
print 'eigenvalues: {0}'.format(eigVals)
print '\neigenvectors: \n{0}'.format(eigVecs)

# Use np.argsort to find the top eigenvector based on the largest eigenvalue
inds = np.argsort(eigVals)[::-1]
topComponent = eigVecs[:,inds[0]] 
print '\ntop principal component: {0}'.format(topComponent)


# TEST Eigendecomposition
def checkBasis(vectors, correct):
    return np.allclose(vectors, correct) or np.allclose(np.negative(vectors), correct)
Test.assertTrue(checkBasis(topComponent, [0.68915649, 0.72461254]),
                'incorrect value for topComponent')


# ** PCA scores**

# Use the topComponent and the data from correlatedData to generate PCA scores
correlatedDataScores = correlatedData.map(lambda x: x.dot(topComponent))
print 'one-dimensional data (first three):\n{0}'.format(np.asarray(correlatedDataScores.take(3)))


# TEST PCA Scores
firstThree = [70.51682806, 69.30622356, 71.13588168]
Test.assertTrue(checkBasis(correlatedDataScores.take(3), firstThree),
                'incorrect value for correlatedDataScores')


# **Write a PCA function and evaluate PCA on sample datasets**

# **PCA function**

def pca(data, k=2):
    """Computes the top `k` principal components, corresponding scores, and all eigenvalues.

    Note:
        All eigenvalues should be returned in sorted order (largest to smallest). `eigh` returns
        each eigenvectors as a column.  This function should also return eigenvectors as columns.

    Args:
        data (RDD of np.ndarray): An `RDD` consisting of NumPy arrays.
        k (int): The number of principal components to return.

    Returns:
        tuple of (np.ndarray, RDD of np.ndarray, np.ndarray): A tuple of (eigenvectors, `RDD` of
            scores, eigenvalues).  Eigenvectors is a multi-dimensional array where the number of
            rows equals the length of the arrays in the input `RDD` and the number of columns equals
            `k`.  The `RDD` of scores has the same number of rows as `data` and consists of arrays
            of length `k`.  Eigenvalues is an array of length d (the number of features).
    """
    covData = estimateCovariance(data)
    eigVal, eigVec = eigh(covData)
    ind = np.argsort(eigVal)[::-1]
    topComp = eigVec[:,ind[0]] 
    scores = data.map(lambda x: x.dot(topComp))
    
    # Return the `k` principal components, `k` scores, and all eigenvalues
    return (eigVec, scores, eigVal) 

# Run pca on correlatedData with k = 2
topComponentsCorrelated, correlatedDataScoresAuto, eigenvaluesCorrelated = pca(correlatedData,2)

# Note that the 1st principal component is in the first column
print 'topComponentsCorrelated: \n{0}'.format(topComponentsCorrelated)
print ('\ncorrelatedDataScoresAuto (first three): \n{0}'
       .format('\n'.join(map(str, correlatedDataScoresAuto.take(3)))))
print '\neigenvaluesCorrelated: \n{0}'.format(eigenvaluesCorrelated)

# Create a higher dimensional test set
pcaTestData = sc.parallelize([np.arange(x, x + 4) for x in np.arange(0, 20, 4)])
componentsTest, testScores, eigenvaluesTest = pca(pcaTestData, 3)

print '\npcaTestData: \n{0}'.format(np.array(pcaTestData.collect()))
print '\ncomponentsTest: \n{0}'.format(componentsTest)
print ('\ntestScores (first three): \n{0}'
       .format('\n'.join(map(str, testScores.take(3)))))
print '\neigenvaluesTest: \n{0}'.format(eigenvaluesTest)


# TEST PCA Function
Test.assertTrue(checkBasis(topComponentsCorrelated.T,
                           [[0.68915649,  0.72461254], [-0.72461254, 0.68915649]]),
                'incorrect value for topComponentsCorrelated')
firstThreeCorrelated = [[70.51682806, 69.30622356, 71.13588168], [1.48305648, 1.5888655, 1.86710679]]
Test.assertTrue(np.allclose(firstThreeCorrelated,
                            np.vstack(np.abs(correlatedDataScoresAuto.take(3))).T),
                'incorrect value for firstThreeCorrelated')
Test.assertTrue(np.allclose(eigenvaluesCorrelated, [1.94345403, 0.13820481]),
                           'incorrect values for eigenvaluesCorrelated')
topComponentsCorrelatedK1, correlatedDataScoresK1, eigenvaluesCorrelatedK1 = pca(correlatedData, 1)
Test.assertTrue(checkBasis(topComponentsCorrelatedK1.T, [0.68915649,  0.72461254]),
               'incorrect value for components when k=1')
Test.assertTrue(np.allclose([70.51682806, 69.30622356, 71.13588168],
                            np.vstack(np.abs(correlatedDataScoresK1.take(3))).T),
                'incorrect value for scores when k=1')
Test.assertTrue(np.allclose(eigenvaluesCorrelatedK1, [1.94345403, 0.13820481]),
                           'incorrect values for eigenvalues when k=1')
Test.assertTrue(checkBasis(componentsTest.T[0], [ .5, .5, .5, .5]),
                'incorrect value for componentsTest')
Test.assertTrue(np.allclose(np.abs(testScores.first()[0]), 3.),
                'incorrect value for testScores')
Test.assertTrue(np.allclose(eigenvaluesTest, [ 128, 0, 0, 0 ]), 'incorrect value for eigenvaluesTest')



