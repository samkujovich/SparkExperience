labVersion = 'cs190_week1_v_1_2'


# ** Math review **

# ** Scalar multiplication: vectors **
# Manually calculate your answer and represent the vector as a list of integers values.
x = [3, -6, 0]
y = [4, 8, 16]


# TEST Scalar multiplication
from test_helper import Test
Test.assertEqualsHashed(x, 'e460f5b87531a2b60e0f55c31b2e49914f779981',
                        'incorrect value for vector x')
Test.assertEqualsHashed(y, 'e2d37ff11427dbac7f833a5a7039c0de5a740b1e',
                        'incorrect value for vector y')


# ** Element-wise multiplication: vectors **
# calculate the element-wise multiplication of two vectors by hand and enter the result in the code cell below.
z = [4, 10, 18]


# TEST Element-wise multiplication
Test.assertEqualsHashed(z, '4b5fe28ee2d274d7e0378bf993e28400f66205c2',
                        'incorrect value for vector z')


# ** Dot product **
# calculate the dot product of two vectors by hand and enter the result in the code cell below.  Note that the dot product is equivalent to performing element-wise multiplication and then summing the result.

# Manually calculate your answer and set the variables to their appropriate integer values.
c1 = -11
c2 = 26


# TEST Dot product
Test.assertEqualsHashed(c1, '8d7a9046b6a6e21d66409ad0849d6ab8aa51007c', 'incorrect value for c1')
Test.assertEqualsHashed(c2, '887309d048beef83ad3eabf2a79a64a389ab1c9f', 'incorrect value for c2')


# ** Matrix multiplication **
# calculate the result of multiplying two matrices together by hand and enter the result in the code cell below.
X = [[22, 28], [49, 64]]
Y = [[1, 2, 3], [2, 4, 6], [3, 6, 9]]


# TEST Matrix multiplication
Test.assertEqualsHashed(X, 'c2ada2598d8a499e5dfb66f27a24f444483cba13',
                        'incorrect value for matrix X')
Test.assertEqualsHashed(Y, 'f985daf651531b7d776523836f3068d4c12e4519',
                        'incorrect value for matrix Y')


# ** NumPy **

# ** Scalar multiplication **
import numpy as np

# Create a numpy array with the values 1, 2, 3
simpleArray = np.array([1,2,3])
# Perform the scalar product of 5 and the numpy array
timesFive = simpleArray * 5
print simpleArray
print timesFive

# TEST Scalar multiplication
Test.assertTrue(np.all(timesFive == [5, 10, 15]), 'incorrect value for timesFive')


# ** Element-wise multiplication and dot product **
# NumPy arrays support both element-wise multiplication and dot product.  Element-wise multiplication occurs automatically when you use the `*` operator to multiply two `ndarray` objects of the same length.
# Create a ndarray based on a range and step size.
u = np.arange(0, 5, .5)
v = np.arange(5, 10, .5)

elementWise = u*v
dotProduct = np.dot(u,v)
print 'u: {0}'.format(u)
print 'v: {0}'.format(v)
print '\nelementWise\n{0}'.format(elementWise)
print '\ndotProduct\n{0}'.format(dotProduct)

# TEST Element-wise multiplication and dot product
Test.assertTrue(np.all(elementWise == [ 0., 2.75, 6., 9.75, 14., 18.75, 24., 29.75, 36., 42.75]),
                'incorrect value for elementWise')
Test.assertEquals(dotProduct, 183.75, 'incorrect value for dotProduct')


# ** Matrix math **
from numpy.linalg import inv

A = np.matrix([[1,2,3,4],[5,6,7,8]])
print 'A:\n{0}'.format(A)
# Print A transpose
print '\nA transpose:\n{0}'.format(A.T)

# Multiply A by A transpose
AAt = A * A.T
print '\nAAt:\n{0}'.format(AAt)

# Invert AAt with np.linalg.inv()
AAtInv = np.linalg.inv(AAt)
print '\nAAtInv:\n{0}'.format(AAtInv)

# Show inverse times matrix equals identity
# We round due to numerical precision
print '\nAAtInv * AAt:\n{0}'.format((AAtInv * AAt).round(4))

# TEST Matrix math
Test.assertTrue(np.all(AAt == np.matrix([[30, 70], [70, 174]])), 'incorrect value for AAt')
Test.assertTrue(np.allclose(AAtInv, np.matrix([[0.54375, -0.21875], [-0.21875, 0.09375]])),
                'incorrect value for AAtInv')


# ** Additional NumPy and Spark linear algebra **

# ** Slices **
features = np.array([1, 2, 3, 4])
print 'features:\n{0}'.format(features)

# The last three elements of features
lastThree = features[-3:]

print '\nlastThree:\n{0}'.format(lastThree)

# TEST Slices
Test.assertTrue(np.all(lastThree == [2, 3, 4]), 'incorrect value for lastThree')


# ** Combining `ndarray` objects **
zeros = np.zeros(8)
ones = np.ones(8)
print 'zeros:\n{0}'.format(zeros)
print '\nones:\n{0}'.format(ones)

zerosThenOnes = np.hstack((zeros, ones))   # A 1 by 16 array
zerosAboveOnes = np.vstack((zeros, ones))  # A 2 by 8 array
 
print '\nzerosThenOnes:\n{0}'.format(zerosThenOnes)
print '\nzerosAboveOnes:\n{0}'.format(zerosAboveOnes)

# TEST Combining ndarray objects
Test.assertTrue(np.all(zerosThenOnes == [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]),
                'incorrect value for zerosThenOnes')
Test.assertTrue(np.all(zerosAboveOnes == [[0,0,0,0,0,0,0,0],[1,1,1,1,1,1,1,1]]),
                'incorrect value for zerosAboveOnes')


# ** PySpark's DenseVector **
from pyspark.mllib.linalg import DenseVector

numpyVector = np.array([-3, -4, 5])
print '\nnumpyVector:\n{0}'.format(numpyVector)

# Create a DenseVector consisting of the values [3.0, 4.0, 5.0]
myDenseVector = DenseVector([3.0, 4.0, 5.0])
# Calculate the dot product between the two vectors.
denseDotProduct = DenseVector.dot(myDenseVector, numpyVector)

print 'myDenseVector:\n{0}'.format(myDenseVector)
print '\ndenseDotProduct:\n{0}'.format(denseDotProduct)

# TEST PySpark's DenseVector
Test.assertTrue(isinstance(myDenseVector, DenseVector), 'myDenseVector is not a DenseVector')
Test.assertTrue(np.allclose(myDenseVector, np.array([3., 4., 5.])),
                'incorrect value for myDenseVector')
Test.assertTrue(np.allclose(denseDotProduct, 0.0), 'incorrect value for denseDotProduct')


# ** Python lambda expressions **

# ** Lambda is an anonymous function **
# Example function
def addS(x):
    return x + 's'
print type(addS)
print addS
print addS('cat')

# As a lambda
addSLambda = lambda x: x + 's'
print type(addSLambda)
print addSLambda
print addSLambda('cat')

multiplyByTen = lambda x: x*10
print multiplyByTen(5)

print '\n', multiplyByTen

# TEST Python lambda expressions
Test.assertEquals(multiplyByTen(10), 100, 'incorrect definition for multiplyByTen')


# ** `lambda' uses fewer steps than `def` **

# Code using def that we will recreate with lambdas
def plus(x, y):
    return x + y

def minus(x, y):
    return x - y

functions = [plus, minus]
print functions[0](4, 5)
print functions[1](4, 5)



# The first function should add two values, while the second function should subtract the second
# value from the first value.
lambdaFunctions = [lambda x,y: x+y ,  lambda x,y: x-y]
print lambdaFunctions[0](4, 5)
print lambdaFunctions[1](4, 5)

# TEST lambda fewer steps than def
Test.assertEquals(lambdaFunctions[0](10, 10), 20, 'incorrect first lambdaFunction')
Test.assertEquals(lambdaFunctions[1](10, 10), 0, 'incorrect second lambdaFunction')


# ** Lambda expression arguments **

# One-parameter function
a1 = lambda x: x[0] + x[1]
a2 = lambda (x0, x1): x0 + x1
print 'a1( (3,4) ) = {0}'.format( a1( (3,4) ) )
print 'a2( (3,4) ) = {0}'.format( a2( (3,4) ) )

# Two-parameter function
b1 = lambda x, y: (x[0] + y[0], x[1] + y[1])
b2 = lambda (x0, x1), (y0, y1): (x0 + y0, x1 + y1)
print '\nb1( (1,2), (3,4) ) = {0}'.format( b1( (1,2), (3,4) ) )
print 'b2( (1,2), (3,4) ) = {0}'.format( b2( (1,2), (3,4) ) )

# Use both syntaxes to create a function that takes in a tuple of two values and swaps their order
# E.g. (1, 2) => (2, 1)
swap1 = lambda x: (x[1],x[0])
swap2 = lambda (x0, x1): (x1,x0)
print 'swap1((1, 2)) = {0}'.format(swap1((1, 2)))
print 'swap2((1, 2)) = {0}'.format(swap2((1, 2)))

# Using either syntax, create a function that takes in a tuple with three values and returns a tuple
# of (2nd value, 3rd value, 1st value).  E.g. (1, 2, 3) => (2, 3, 1)
swapOrder = lambda (x0, x1, x2): (x1, x2, x0)
print 'swapOrder((1, 2, 3)) = {0}'.format(swapOrder((1, 2, 3)))

# create a function that takes in three tuples each with two values.  The
# function should return a tuple with the values in the first position summed and the values in the
# second position summed. E.g. (1, 2), (3, 4), (5, 6) => (1 + 3 + 5, 2 + 4 + 6) => (9, 12)
sumThree = lambda (x0,x1),(y0,y1),(z0,z1):(x0+y0+z0,x1+y1+z1)
print 'sumThree((1, 2), (3, 4), (5, 6)) = {0}'.format(sumThree((1, 2), (3, 4), (5, 6)))

# TEST Lambda expression arguments
Test.assertEquals(swap1((1, 2)), (2, 1), 'incorrect definition for swap1')
Test.assertEquals(swap2((1, 2)), (2, 1), 'incorrect definition for swap2')
Test.assertEquals(swapOrder((1, 2, 3)), (2, 3, 1), 'incorrect definition fo swapOrder')
Test.assertEquals(sumThree((1, 2), (3, 4), (5, 6)), (9, 12), 'incorrect definition for sumThree')


# ** Restrictions on lambda expressions **

# This code will fail with a syntax error, as we can't use print in a lambda expression
import traceback
try:
    exec "lambda x: print x"
except:
    traceback.print_exc()


# ** Functional programming **

# Create a class to give our examples the same syntax as PySpark
class FunctionalWrapper(object):
    def __init__(self, data):
        self.data = data
    def map(self, function):
        """Call `map` on the items in `data` using the provided `function`"""
        return FunctionalWrapper(map(function, self.data))
    def reduce(self, function):
        """Call `reduce` on the items in `data` using the provided `function`"""
        return reduce(function, self.data)
    def filter(self, function):
        """Call `filter` on the items in `data` using the provided `function`"""
        return FunctionalWrapper(filter(function, self.data))
    def __eq__(self, other):
        return (isinstance(other, self.__class__)
            and self.__dict__ == other.__dict__)
    def __getattr__(self, name):  return getattr(self.data, name)
    def __getitem__(self, k):  return self.data.__getitem__(k)
    def __repr__(self):  return 'FunctionalWrapper({0})'.format(repr(self.data))
    def __str__(self):  return 'FunctionalWrapper({0})'.format(str(self.data))

# Map example

# Create some data
mapData = FunctionalWrapper(range(5))

# Define a function to be applied to each element
f = lambda x: x + 3

# Imperative programming: loop through and create a new object by applying f
mapResult = FunctionalWrapper([])  # Initialize the result
for element in mapData:
    mapResult.append(f(element))  # Apply f and save the new value
print 'Result from for loop: {0}'.format(mapResult)

# Functional programming: use map rather than a for loop
print 'Result from map call: {0}'.format(mapData.map(f))

from operator import add
dataset = FunctionalWrapper(range(10))

# Multiply each element by 5
mapResult = dataset.map(lambda x: x*5)
# Keep the even elements
filterResult = dataset.filter(lambda x: x%2 ==0)
# Sum the elements
reduceResult = dataset.reduce(add)

print 'mapResult: {0}'.format(mapResult)
print '\nfilterResult: {0}'.format(filterResult)
print '\nreduceResult: {0}'.format(reduceResult)

# TEST Functional programming
Test.assertEquals(mapResult, FunctionalWrapper([0, 5, 10, 15, 20, 25, 30, 35, 40, 45]),
                  'incorrect value for mapResult')
Test.assertEquals(filterResult, FunctionalWrapper([0, 2, 4, 6, 8]),
                  'incorrect value for filterResult')
Test.assertEquals(reduceResult, 45, 'incorrect value for reduceResult')


# ** Composability **
# Since our methods for map and filter in the `FunctionalWrapper` class return `FunctionalWrapper` objects, we can compose (or chain) together our function calls.
(dataset
 .map(lambda x: x + 2)
 .reduce(lambda x, y: x * y))

from operator import add

# Multiply the elements in dataset by five, keep just the even values, and sum those values
finalSum = (dataset.map(lambda x: x*5).filter(lambda x: x%2==0).reduce(add))
print finalSum

# TEST Composability
Test.assertEquals(finalSum, 100, 'incorrect value for finalSum')


# ** CTR data download **

from IPython.lib.display import IFrame

IFrame("http://labs.criteo.com/downloads/2014-kaggle-display-advertising-challenge-dataset/",
       600, 350)

import glob
import os.path
import tarfile
import urllib
import urlparse

url = 'http://labs.criteo.com/wp-content/uploads/2015/04/dac_sample.tar.gz'

url = url.strip()
baseDir = os.path.join('data')
inputPath = os.path.join('cs190', 'dac_sample.txt')
fileName = os.path.join(baseDir, inputPath)
inputDir = os.path.split(fileName)[0]

def extractTar(check = False):
    # Find the zipped archive and extract the dataset
    tars = glob.glob('dac_sample*.tar.gz*')
    if check and len(tars) == 0:
      return False

    if len(tars) > 0:
        try:
            tarFile = tarfile.open(tars[0])
        except tarfile.ReadError:
            if not check:
                print 'Unable to open tar.gz file.  Check your URL.'
            return False

        tarFile.extract('dac_sample.txt', path=inputDir)
        print 'Successfully extracted: dac_sample.txt'
        return True
    else:
        print 'You need to retry the download with the correct url.'
        print ('Alternatively, you can upload the dac_sample.tar.gz file to your Jupyter root ' +
              'directory')
        return False


if os.path.isfile(fileName):
    print 'File is already available. Nothing to do.'
elif extractTar(check = True):
    print 'tar.gz file was already available.'
elif not url.endswith('dac_sample.tar.gz'):
    print 'Check your download url.  Are you downloading the Sample dataset?'
else:
    # Download the file and store it in the same directory as this notebook
    try:
        urllib.urlretrieve(url, os.path.basename(urlparse.urlsplit(url).path))
    except IOError:
        print 'Unable to download and store: {0}'.format(url)

    extractTar()


import os.path
baseDir = os.path.join('data')
inputPath = os.path.join('cs190', 'dac_sample.txt')
fileName = os.path.join(baseDir, inputPath)

if os.path.isfile(fileName):
    rawData = (sc
               .textFile(fileName, 2)
               .map(lambda x: x.replace('\t', ',')))  # work with either ',' or '\t' separated data

print rawData.take(1)
rawDataCount = rawData.count()
print rawDataCount
# This line tests that the correct number of observations have been loaded
assert rawDataCount == 100000, 'incorrect count for rawData'
if rawDataCount == 100000:
    print 'Criteo data loaded successfully!'
