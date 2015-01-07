#! /usr/bin/python

"""
ann.py

Steve Wilson
August 2013

An adaptive neural network training framework for SAR image chip feature vectors.
Can be run from the command line with the top level directory of the dataset as an argument.
e.g.
    python ann.py oilSlickClassification/chips/
"""
import sys
import os

import cv2
import numpy as np

# read the features file into a dictionary
# ignores the contours
def read_features(path):
    f = open(path,'r')
    lines = f.readlines()
    d = {k.strip():eval(v.strip()) for (k,v) in [tuple(line.split(':')) for line in lines if ":" in line] if v.strip()!=""}
    f.close()
    return d

# finds all the features.txt files in the directory structure
# note that the class_dirs should be the 
def find_features(top_level_dir):
    print "looking for feature files in:",top_level_dir
    feature_vecs = {}
    if not os.path.isdir(top_level_dir):
        usage()
    else:
        for class_dir in os.listdir(top_level_dir):
            feature_vecs[class_dir] = []
            if os.path.isdir(top_level_dir + os.sep + class_dir):
                for dirname,dirs,files in os.walk(top_level_dir + os.sep + class_dir):
                    for fname in files:
                        if fname == "features.txt":
                            feature_vecs[class_dir].append(read_features(dirname+os.sep+fname))
            else:
                print "not a dir:",class_dir
    return feature_vecs

# get a list form of the dictionary values using keys from 'keys'
# this is a little different than d.values() since we are getting the values in a set order
def convert_to_vec(d,keys):
    return [d[k] for k in keys]

# generate an ann, then train and test it
def generate(features):

    # list of all of the dicts from feature files
    feature_dicts = []
    # parallel list of the classifications of the chips represented by the dicts
    class_list = []
    # list of all classes
    class_names = []
    # list of keys so features can be referenced later
    feature_names = []

    # features is a dict in the form:
    #   {class1:{feature_dict},{feature_dict}... class2:{feature_dict}...}
    for clss,dicts in features.items():
        class_names.append(clss)
        for d in dicts:
            feature_dicts.append(d)
            class_list.append(clss)
            # first time only- get feature names
            if not feature_names:
                feature_names = list(d.keys())
    # make sure parallel lists were same length, else an error will be thrown
    assert len(feature_dicts) == len(class_list)
    # make sure we got the feature names
    assert feature_names != None
    # create a matrix of all inputs
    input_vec = [convert_to_vec(d,feature_names) for d in feature_dicts]
    # create the corresponding output matrix
    output_vec = [class_names.index(x) for x in class_list]
    print "classes are:",",".join(class_names)

    # for tracking and calculating stats
    total = 0
    total_err = 0
    count = len(input_vec)
    # perform leave-one-out testing
    # i is the index of the left-out vector
    for i in range(count):
        print ""
        print "leave one out validation:",i+1,"of",count
        train_inputs = np.array(input_vec[:i] + input_vec[i+1:])
        train_outputs = np.array(output_vec[:i] + output_vec[i+1:],dtype=np.float32)
        test_inputs = np.array([input_vec[i]])
        test_outputs = np.array([output_vec[i]])
        result,error = train_and_test((train_inputs,train_outputs),(test_inputs,test_outputs))
        total += result
        total_err += error
    print ""
    print "accuracy:",100*float(total)/count,"%"
    print "error rate:",100*float(total_err)/count,"%"
    # TODO save the ANN to a file, then read in from classify.py

def train_and_test(training,testing):
    # create ann object
    ann = cv2.ANN_MLP()
    ann.create(np.array([len(training[0][0]),8,8,1]))

    # make output into an 1xN matrix instead of Nx1
    output_col = training[1][np.newaxis].T
    # make sure inputs and outputs are same size
    assert len(training[1]) == len(training[0])
    # train the ann
    ann.train(training[0],output_col,np.array([1]*len(training[0]),dtype=np.float32))
    # test the ann
    retval,outputs = ann.predict(testing[0])
    print "results:",outputs
    print "actual:",testing[1]
    assert len(outputs) == len(testing[1])
    correct = 0
    error = 0
    # calculate error
    for i in range(len(outputs)):
        if round(outputs[i],0) == testing[1][i]:
            correct += 1
        error += abs(testing[1][i]-outputs[i])
    print "correct:",correct
    return correct,error

def usage():
    print "usage: python ann.py TOP_LEVEL_DIR"

if __name__ == "__main__":
    args = sys.argv
    if len(args) != 2:
        usage()
    else:
        generate(find_features(args[1].rstrip(os.sep)))
