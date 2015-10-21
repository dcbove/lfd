'''
Created on Oct 20, 2015

@author: dan
'''
import numpy as np
import random
import sys


# this is the true function we are attempting to model
# x is an array of size d (d = 3)
# w0 = bias (1).  w1 = -3.  w2 = 4
def f(x):
    bias = 1
    value = (bias * x[0]) + ((-3) * x[1]) + ((4) * x[2])
    return sign(value)

# test function
def h(w, x):
    a = np.array(w)
    b = np.array(x)
    c = np.dot(a, b)
    return sign(c)
    
# return 1 if greater than 0, -1 if less than, otherwise 0
def sign(x):
    if (x > 0): 
        return 1
    if (x < 0): 
        return -1
    return 0
    
# try 50 times to find a misclassified example.  that'd be one where h(w, x) returns the wrong results for the current w
def find_misclassifed_example(w, td):
    attempt = 0
    while (attempt < 50):
        i = random.randint(0, trainingDataCount-1)
        sampleData = td[i]
        testResult = h(w, sampleData)
        trainingDataResult = f(sampleData)
        if (testResult != trainingDataResult):
            return sampleData
        attempt += 1
    
    return None

# given the training data, the known f, and current w, return % that match
def calc_match_rate(trainingData, w):
    match = 0
    for td in trainingData:
        if (f(td) == h(w, td)):
            match += 1
            
    return match*100/trainingDataCount

# training data
trainingDataCount = 100
trainingData = []
for i in range(trainingDataCount):
    datum = np.array([1, random.randint(-10, 10), random.randint(-10, 10)])
    trainingData.append(datum)
    print("Training Datum {}, Sign {}".format(datum, f(datum)))
        
# opening value for w
w = np.array([0, 0, 0])

# training iterations
t_iters = 100

# train
for t in range(t_iters):
    misclassified_example = find_misclassifed_example(w, trainingData)
    if (misclassified_example == None):
        print('found success. w={}, Match Rate={} '.format(w, calc_match_rate(trainingData, w)))
        # re-examine training
        print("training data with ", w)
        for td in trainingData:
            print("td {}\t f {}\t h {}\tmatch={}".format(td, f(td), h(w, td), f(td)==h(w,td)))    
        sys.exit()
    else:
        y = f(misclassified_example)
        print('step {}, w={}, y={}, mce={}, u={}, new w={}, Match Rate={}'.format(t, w, y, misclassified_example, (y*misclassified_example), np.add(w, y*misclassified_example), calc_match_rate(trainingData, w)))
        w = w + (y * misclassified_example)





    
