#! /usr/bin/python

"""
feature_funcs.py
-------------
Description: Function libary for calculating features from region objects
Author:      Steve Wilson
Created:     July 16, 2013

Instructions:
1) add any feature functions desired
2) decorate functions to use with @use_function
3) set parameter of @use_function to True if it should be used in feature vector
4) call the feature_funcs method from outside files to get a dictionary of the functions

NOTE: in order to use ensure_feature(), the feature to be ensured ,must be decorated with use_function, but the parameter does not have to be True
"""


import cv2
import math
import numpy as np

DEBUG = False
# set true to convert all instances of infinity to zero when returning feature values
INF_TO_ZERO = True

"""
Class to keep track of feature functions
Also allows for memoization of feature values in case they are used in muliple calculations
"""
class feature_function:
    funcs = {}
    def __init__(self,func,remember):
        self.f = func
        self.name = func.__name__
        if remember:
            feature_function.funcs[func.__name__] = self
        if DEBUG:
            print "using feature:",func.__name__

    def __call__(self,region):
        # check if feature has been computed already
        result = region.features.get(self.name)
        # check for none since 0 would also be considered True if trying !result
        if result==None:
            if DEBUG:
                print "calculating",self.name+"...",
            result = self.f(region)
            if DEBUG:
                print "...done",result
            # do not put inf,-inf, or NaN into feature vectors
            # use 0 instead
            try:
                if INF_TO_ZERO and (result == float('inf') or math.isnan(result) or result == float("-inf")):
                    result = 0
            except:
                try:
                    if INF_TO_ZERO and (np.isinf(result).any() or np.isnan(result).any or np.isneginf(result).any()):
                        return np.zeros(result.shape)
                except Exception as e:
                    print "could not check for inf or nan in object:",type(result),e
                    return 0
                    
        return result

"""
Decorator function that will turn functions into feature_function objects and pass in the do_use parameter
"""
def use_function(do_use):
    def ffwrapper(f):
        return feature_function(f,do_use)
    return ffwrapper

@use_function(False)
def gradient(region):
    xgrad = cv2.Scharr(region.mat,cv2.CV_16S,1,0)
    ygrad = cv2.Scharr(region.mat,cv2.CV_16S,0,1)
#    cv2.imshow("xgrad",xgrad)
#    cv2.waitKey(0)
#    abs_xgrad = cv2.convertScaleAbs(xgrad)
#    cv2.imshow("absx",abs_xgrad)
#    cv2.waitKey(0)
#    cv2.imshow("ygrad",ygrad)
#    cv2.waitKey(0)
#    abs_ygrad = cv2.convertScaleAbs(ygrad)
#    cv2.imshow("absy",abs_ygrad)
#    cv2.waitKey(0)
    gradient_image = cv2.addWeighted(xgrad,.5,ygrad,.5,0,dtype=cv2.CV_16S)
#    cv2.imshow("gradient",gradient_image)
#    cv2.waitKey(0)
    contour_image = np.zeros(region.mat.shape,dtype=np.uint8)
    cv2.drawContours(contour_image,region.contours,-1,255,1)
#    cv2.imshow("contours",contour_image)
#    cv2.waitKey(0)
    contour_points = np.nonzero(contour_image)
    if len(region.contours) > 0:
        return np.array([gradient_image[x,y] for x,y in zip(contour_points[0],contour_points[1])])
    else:
        # need to return something in order to calculate statistics
        return np.array([0])

# not currently used, but may be useful for calculating other features
@use_function(False)
def perimeter(region):
    # handles 2 different versions of the region object
    if hasattr(region,'coords'):
        return abs(cv2.arcLength(region.coords,True))
    else:
        return sum([abs(cv2.arcLength(c,True)) for c in region.contours])

@use_function(False)
def area(region):
    # handles 2 different versions of the region object
    if hasattr(region,'coords'):
        return abs(cv2.contourArea(region.coords))
    else:
        return sum([abs(cv2.contourArea(c)) for c in region.contours])

@use_function(True)
def obj_mean(region):
    return cv2.mean(region.mat,mask=region.mask)[0]

@use_function(True)
def obj_stddev(region):
    return cv2.meanStdDev(region.mat,mask=region.mask)[1][0][0]

@use_function(False)
def entropy(region):
    flat = region.mat[zip(region.inside_points)].flatten()
    counts = np.bincount(flat)
    num_vals = len(flat)
    e = 0
    for count in counts:
        if count:
            p = count/float(num_vals)
            if p:
                e -= p * math.log(p,2)
    return e

@use_function(True)
def obj_pmr(region):
    ensure_feature(region,obj_mean)
    ensure_feature(region,obj_stddev)
    return region.features['obj_mean']/region.features['obj_stddev']

@use_function(True)
def bgnd_mean(region):
    return cv2.mean(region.mat,mask=~region.mask)[0]

@use_function(True)
def bgnd_stddev(region):
    return cv2.meanStdDev(region.mat,mask=~region.mask)[1][0][0]

@use_function(True)
def bgnd_pmr(region):
    ensure_feature(region,bgnd_mean)
    ensure_feature(region,bgnd_stddev)
    return region.features['bgnd_mean']/region.features['bgnd_stddev']

@use_function(True)
def mean_contrast(region):
    ensure_feature(region,obj_mean)
    ensure_feature(region,bgnd_mean)
    return abs(region.features['bgnd_mean']-region.features['obj_mean'])

@use_function(True)
def pmr_ratio(region):
    ensure_feature(region,obj_pmr)
    ensure_feature(region,bgnd_pmr)
    return region.features['obj_pmr']/region.features['bgnd_pmr']

@use_function(True)
def intensity_ratio(region):
    ensure_feature(region,obj_mean)
    ensure_feature(region,bgnd_mean)
    return abs(region.features['obj_mean']/region.features['bgnd_mean'])

@use_function(True)
def stddev_ratio(region):
    ensure_feature(region,obj_stddev)
    ensure_feature(region,bgnd_stddev)
    return abs(region.features['obj_stddev']/region.features['bgnd_stddev'])

@use_function(True)
def mean_gradient(region):
    ensure_feature(region,gradient)
    return np.mean(region.features['gradient'])

@use_function(True)
def max_gradient(region):
    ensure_feature(region,gradient)
    return np.max(region.features['gradient'])

@use_function(True)
def gradient_stddev(region):
    ensure_feature(region,gradient)
    return np.std(region.features['gradient'])

# TODO implement this and any others desired
@use_function(False)
def asymmetry(region):
    pass

def feature_dict():
    """
    Returns any functions that a are decorated by @use_function(True)
    """
    return feature_function.funcs.items()

def labels():
    """
    Returns just the keys of the feature functions
    >>> [x for x in labels()][:4]
    ['area', 'obj_mean', 'obj_stddev', 'entropy']
    """
    return feature_function.funcs.keys()

def ensure_feature(region,func):
    # Check if None since get() could return 0 and still be valid
    if region.features.get(func.name)==None:
        region.features[func.name] = func(region) 
