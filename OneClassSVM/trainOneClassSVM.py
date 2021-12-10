import numpy as np
from numpy import genfromtxt

from sklearn.svm import OneClassSVM

import _pickle as cPickle

# Data
num_joint = 7
num_seq = 10

train_data = genfromtxt('./data/TrainingData.csv', delimiter=',')

clf = OneClassSVM(gamma=0.1, nu=0.001).fit(abs(train_data))

# save the classifier
with open('./model/ocsvm_residual.pkl', 'wb') as fid:
    cPickle.dump(clf, fid)    

print("Training Finished!")


# Data
test_data = genfromtxt('./data/TestingData.csv', delimiter=',')

residual_train = abs(train_data)
residual_test = abs(test_data)

np.savetxt('./result/training_result.txt',clf.decision_function(residual_train))
np.savetxt('./result/testing_result.txt',clf.decision_function(residual_test))

# External torque Response
resolution = 0.01
ext_array = np.arange(0,3,resolution)
result = np.empty((ext_array.shape[0], 2), float)

for idx, ext in enumerate(ext_array):
    user_test_data = ext*np.abs(np.ones((1,150)))
    output = clf.decision_function(user_test_data)
    result[idx] = np.array([ext, output[0]])

np.savetxt('./result/response.txt', result)

################################################################
random_idx = np.random.randint(test_data.shape[0])
user_test_data = residual_test[random_idx]
user_test_data[135] += 0.5
print(clf.decision_function(np.array([user_test_data])))