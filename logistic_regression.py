import csv
import numpy as np
import math 
import copy
import random

def read_data(path): 
    with open(path, 'r') as f:
        data = csv.reader(f, delimiter=',')        
        training_inputs = [] 
        for i, line in enumerate(data):
            if i == 0: #first line is title
                continue 
            else:
                feature_vector = [1] #account for bias term 
                feature_vector.extend(line[:-1])
                feature_vector = np.array(feature_vector).astype('float64')
                label = line[-1] 
                training_inputs.append((feature_vector, label))
        return training_inputs

def compute_full_model(data, epoch): #finds all weights using 1 vs all
    labels = []
    for line in data:
        if line[-1] not in labels:
            labels.append(line[-1])
    models = [] 
    for label in labels:
        models.append(train_model(epoch, data, label))
    return (models, labels)

def train_model(epoch, data, postive_label): #1 vs all 
    weights = np.zeros(len(data[0][0])) #data is [(feature_vector, label),...]
    learner_rate = 0.1 #learning rate for stochastic gradient decent 
    decay = 0.99
    data = copy.copy(data) #prevent aliasing issues
    for _ in range(epoch):
        learner_rate = learner_rate*decay
        random.shuffle(data)
        for feature_vector, label in data:  #implement shuffle 
            gradient_vector = compute_graident(weights, feature_vector, label, postive_label)*learner_rate*feature_vector #formula
            weights = weights + gradient_vector
    return weights 

def compute_graident(weights, feature_vector, label, postive_label):
    if label == postive_label: y = 1
    else: y = 0
    assert(len(feature_vector)   == len(weights))
    dot_product = np.dot(weights, feature_vector)
    if dot_product > 600: #so exp function doesn't overflow 
        return y - 1
    elif dot_product < -600:
        return y 
    exp_val = math.exp(dot_product)
    return y - ((exp_val)/(1+exp_val))



def predict(models, labels, feature_vector):
    probability_list = []
    assert(len(models) == len(labels))
    for model, label in zip(models, labels):
        dot_product = np.dot(model, feature_vector)
        if dot_product > 400: #so exp function doesn't overflow 
            probability = 0
        elif dot_product < -400:
            probability = 1
        else:
            exp_val = math.exp(dot_product)
            probability = exp_val/(1+exp_val)
        probability_list.append((probability, label))
    return max(probability_list, key = lambda probs: probs[0])[1]

def training_error(training_set, models, labels):
    error = 0
    for feature_vector, label in training_set:
        if predict(models, labels, feature_vector) != label:
            error += 1
    return error/len(training_set)

data = read_data("iris_full_set.csv")
models, labels = compute_full_model(data, 200)

print(training_error(data,models,labels))

#predicted probablity of each one
#weights of each flower
#train error 
#test error 