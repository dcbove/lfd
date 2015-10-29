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
# this is the true function we are attempting to model
bias = 14
w1 = 0.5
w2 = 0.25

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
    
# try to find a misclassified example.  that'd be one where h(w, x) returns the wrong results for the current w
def find_misclassifed_example(w, df):
    while (True):
        i = random.randint(0, training_data_count - 1)
        datum = df.ix[i]
        training_data_result = datum.training_value
        sample_data = [datum.w0, datum.w1, datum.w2]
        testResult = h(w, sample_data)
        if (testResult != training_data_result):
            return np.array(sample_data)

# given the training data, the known f, and current w, return % that match
def calc_match_rate(dfTraining, w):
    training_matrix = dfTraining.ix[:, ['w0', 'w1', 'w2']]
    result = np.dot(w, training_matrix.T)
    signed_result = np.array([sign(x) for x in result])
    
    training_values = dfTraining.ix[:, ['training_value']]
    
    # determine how many elements of signed_result and training_values are the same
    tv = np.squeeze(np.asarray(training_values.T))
    equal_count = len(signed_result[signed_result - tv == 0])
    return (equal_count * 100) / training_data_count

# training data
training_data_count = 1000
df_training = pd.DataFrame(index=np.arange(0, training_data_count), columns=['w0', 'w1', 'w2', 'training_value'])
for i in range(training_data_count):
    w = [1, random.randint(-boundxy, boundxy), random.randint(-boundxy, boundxy)]
    trainingValue = f(w)
    df_training.loc[i] = [w[0], w[1], w[2], trainingValue]

# opening value for w
w = np.array([1, 1, 1])

# training iterations
t_iters = 10000

# train
match_history = []
stopwatch = datetime.now()
for t in range(t_iters):
    match_rate = calc_match_rate(df_training, w)
    match_history.append(match_rate)
    if match_rate > 98:
        print('found success. w={}, Match Rate={} '.format(w, match_rate))
        break
    
    
    if t % 50 == 0:
        print("iter: {}, timediff: {}.{}".format(t, (datetime.now() - stopwatch).seconds, (datetime.now() - stopwatch).microseconds))
        stopwatch = datetime.now()
    
    misclassified_example = find_misclassifed_example(w, df_training)
    if (misclassified_example == None):
        print('could not find misclassified example. w={}, Match Rate={} '.format(w, match_rate))
        break
    else:
        y = f(misclassified_example)
        w = w + (y * misclassified_example)

# plot training data
training_values = np.squeeze(np.array(df_training.ix[:, ['training_value']]))
training_xs = np.squeeze(np.array(df_training.ix[:, ['w1']].T))
training_ys = np.squeeze(np.array(df_training.ix[:, ['w2']].T))

f, (ax1, ax2) = plt.subplots(1, 2)

color = [str((x + 1) / 2) for x in training_values]
plt.axhline(0, color='gray')
plt.axvline(0, color='gray')
ax1.scatter(training_xs, training_ys, c=color)

# for w, determine x2 at x1=10, x1=-10
leftx2 = (-w[0] - w[1] * (-boundxy)) / w[2]
rightx2 = (-w[0] - w[1] * (boundxy)) / w[2]
l = Line2D([-boundxy, boundxy], [leftx2, rightx2])                                    
ax1.add_line(l)

# plot training match history
series = pd.Series(match_history, index=np.arange(0, len(match_history)))
series.plot()
 
plt.show()


    
