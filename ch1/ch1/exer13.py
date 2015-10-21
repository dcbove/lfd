'''
Created on Oct 20, 2015

@author: dan
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import random
from datetime import datetime

# plot bounds - its a square
boundxy = 50

# build true function f
bias = 14
w1 = 0.5
w2 = 0.25

# this is the true function we are attempting to model
trueW = np.array([-bias, w1, w2])
def f(x):
    value = np.dot(trueW, x)
    sgn = sign(value)
    return sgn

# test function
def h(w, x):
    c = np.dot(w, x)
    return sign(c)
    
# return 1 if greater than 0, -1 if less than, otherwise 0
def sign(x):
    if (x > 0): 
        return 1
    elif (x < 0): 
        return -1
    else:
        return 0
    
# try 50 times to find a misclassified example.  that'd be one where h(w, x) returns the wrong results for the current w
def find_misclassifed_example(w, td):
    while (True):
        i = random.randint(0, trainingDataCount - 1)
        sampleData = td[i]
        testResult = h(w, sampleData)
        trainingDataResult = f(sampleData)
        if (testResult != trainingDataResult):
            return sampleData

# given the training data, the known f, and current w, return % that match
def calc_match_rate(trainingData, w):
    match = 0
    for td in trainingData:
        if (f(td) == h(w, td)):
            match += 1
            
    return match * 100 / trainingDataCount

# training data
trainingDataCount = 1000
trainingData = []
trainingValues = []
for i in range(trainingDataCount):
    datum = [1, random.randint(-boundxy, boundxy), random.randint(-boundxy, boundxy)]
    trainingData.append(datum)
    trainingValues.append(f(datum))
    # print("Training Datum {}, Sign {}".format(datum, f(datum)))
trainingData = np.array(trainingData)
trainingValues = np.array(trainingValues)

# opening value for w
w = np.array([0, 0, 0])

# training iterations
t_iters = 10000

# train
matchHistory = []
stopwatch = datetime.now()
for t in range(t_iters):
    match_rate = calc_match_rate(trainingData, w)
    if match_rate > 98:
        print('found success. w={}, Match Rate={} '.format(w, match_rate))
        break
    
    matchHistory.append(match_rate)
    
    if t % 50 == 0:
        print("iter: {}, timediff: {}.{}".format(t, (datetime.now()-stopwatch).seconds, (datetime.now()-stopwatch).microseconds))
        stopwatch = datetime.now()
    
    misclassified_example = find_misclassifed_example(w, trainingData)
    if (misclassified_example == None):
        print('could not find misclassified example. w={}, Match Rate={} '.format(w, match_rate))
        break
    else:
        y = f(misclassified_example)
        # print('step {}, w={}, y={}, mce={}, u={}, new w={}, Match Rate={}'.format(t, w, y, misclassified_example, (y * misclassified_example), np.add(w, y * misclassified_example), calc_match_rate(trainingData, w)))
        w = w + (y * misclassified_example)

print("final w ", w, match_rate)

# plot training data
f, (ax1, ax2) = plt.subplots(1, 2)
color = [str((x + 1) / 2) for x in trainingValues]
plt.axhline(0, color='gray')
plt.axvline(0, color='gray')
ax1.scatter(trainingData.T[1], trainingData.T[2], c=color)

# for w, determine x2 at x1=10, x1=-10
leftx2 = (-w[0] - w[1] * (-boundxy)) / w[2]
rightx2 = (-w[0] - w[1] * (boundxy)) / w[2]
l = Line2D([-boundxy, boundxy], [leftx2, rightx2])                                    
ax1.add_line(l)

# plot training match history
series = pd.Series(matchHistory, index=np.arange(0, len(matchHistory)))
series.plot()
 
plt.show()


    
