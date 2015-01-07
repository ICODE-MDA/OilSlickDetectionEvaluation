
#! /usr/bin/python

"""
classify.py
-------
Description: Locates dark spots and decides if they are oil spills or not
Author:      Steve Wilson
Created:     June 25, 2013
"""

import random
import math
import pickle
import os

import numpy as np
import cv2
from osgeo import gdal
from scipy import cluster

from sar_image import *
import feature_funcs

# don't even consider dark regions less than MINSIZE pixels
MINSIZE = 100

def bounds_check(rect,dims):
    x,y,w,h = rect
    xmax,ymax = dims
    x,y = max(0,x),max(0,y)
    if w+x > xmax:
        w = xmax-x
    if y+h > ymax:
        h = ymax-y
    return (x,y,w,h)

def expand_rect(rect,xin,yin):
    x,y,w,h = rect
    return (x-(xin/2),y-(yin/2),w+xin,h+yin)

class Region:
    def __init__(self,contour):
        self.coords = contour
        self.features = {}
        self.is_oil = False
        self.mat = None
        self.mask = None
        self.inside_points = None
        self.offset = None

    def setup_mat(self,image,xincrease=20,yincrease=20):
        x,y,w,h = bounds_check( expand_rect(tuple(cv2.boundingRect(self.coords)),xincrease,yincrease), image.get_dimensions() )
        self.mat = image.to_array(x,y,w,h)
        self.offset = (x,y)
        self.mask = np.zeros(self.mat.shape,np.uint8)
        cv2.drawContours(self.mask,[self.localized_coords()],0,255,-1)
        self.inside_points = np.nonzero(self.mask)
        # debug
        #self.display()

    def cleanup(self):
        self.mat = None
        self.mask = None
        self.inside_points = None

    def calc_features(self):
        all_features = {name:func(self) for name,func in feature_funcs.feature_dict()}
        # this only retains features that were actually decorated
        # all_features will also contain any features used for calculations
        self.features = {k:v for (k,v) in all_features.items() if k in feature_funcs.labels()}

    def to_vector(self):
        return [self.features[L] for L in feature_funcs.labels()]

    def localized_coords(self):
        return self.coords - self.offset

    def display(self):
        disp_mat = self.mat.copy()
        cv2.drawContours(disp_mat,[self.localized_coords()],-1,255,3)
        cv2.imshow("ROI",disp_mat*25)
        cv2.waitKey(0)

    def save(self,data_dir,name):
        path = data_dir + os.sep + name
        image_path = path + ".png"
        pickle_path = path + ".pkl"
        cv2.imwrite(image_path,self.mat)
        with open(pickle_path,'w') as pickle_file:
            pickle.dump(self,pickle_file)

def sort_pixels(image,shape):
    spill_pixels = []
    outside_pixels = []
#    test_image = np.zeros(image.shape)
#    cv2.fillConvexPoly(test_image,shape,1)
    border = 50
    x,y,w,h = cv2.boundingRect(shape)
    x = max(0,x-border)
    y = max(0,y-border)
    w += 2*border
    h += 2*border
    x2 = min(x+w,image.shape[1])
    y2 = min(y+h,image.shape[0])
    ### Testing
#    rect = test_image[y:y+h,x:x+w]
#    cv2.imshow("test",rect)
#    cv2.waitKey(0)
    ### Testing
    for i in range(x,x2):
        for j in range(y,y2):
#            if test_image[j,i]:
            dist = cv2.pointPolygonTest(shape,(i,j),True)
            if dist >= 0:
                spill_pixels.append(image[j,i])
            elif dist >= -50:
                outside_pixels.append(image[j,i])
                
#    print cv2.contourArea(shape),len(pixels),cv2.arcLength(shape,True)
#    assert cv2.contourArea(shape) == len(pixels)
    return np.array(spill_pixels),np.array(outside_pixels)

# old implementation trying to just use entropy calculation to classify
# this method was not robust
def decide(image,shape):
    spill_pix,out_pix = sort_pixels(image,shape)
    spill_entropy = entropy(spill_pix)
    out_entropy = entropy(out_pix)
    total = spill_entropy / out_entropy
    print "Entropy:", spill_entropy,"/",out_entropy,"=",total
    if .55 < total < .67:
        return True,total
    else:
        return False,total

def stretch_value(x,new_max,new_min,old_max,old_min):
    ratio = float((new_max-new_min))/(old_max-old_min)
    result = (ratio*(x-old_min))+new_min
    return int(min(result,new_max))

def fit_to_8bit(mat):
    x = np.max(mat)
    n = np.min(mat)
    m = np.mean(mat)
    s = np.std(mat)
    fit_vector = np.vectorize(stretch_value,otypes=[np.uint8])
    return fit_vector(mat,255,0,m+3*s,n)

# TODO: train an actual robust classifier
# kmeans will have extremely low accuracy
def classify(regions,k):
    centroids, labels = cluster.vq.kmeans2(np.array([r.to_vector() for r in regions]),k,minit='points')
    return (centroids, labels)

def classify_all(src,contour_list,train=True):

    print "Classifying..."
    image = Sar_Image(src)

    print "number of contours:",len(contour_list)
    potential_spill_regions = [Region(x) for x in contour_list if cv2.contourArea(x) > MINSIZE]
    print "potential spill regions:",len(potential_spill_regions)


    data_dir = "data"
    region_name_base = "region"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for num,region in enumerate(potential_spill_regions):
#        print "setting up neighborhood matrix..."
        region.setup_mat(image)
        if train:
            region.save(data_dir,region_name_base+"_"+str(num))
        else:
            region.calc_features()
#        print "cleaning up region data"
        region.cleanup()

    if train:
        print "chipping done"
        sys.exit()

    print "pickling feature vectors"
    # maybe export feature vector so that it doesn't need to be recalculated while testing classifiers
    with open("region_vectors.pkl",'w') as pklfile:
        pickle.dump(potential_spill_regions,pklfile)

    centroids, labels = classify(potential_spill_regions,8)
    print [x for x in feature_funcs.labels()]
    print "centroids:"
    for c in centroids:
        print c

    result_image = cv2.cvtColor(fit_to_8bit(image.to_array()),cv2.cv.CV_GRAY2RGB)

    print "Saving result image..."

    result_loc = "detections.jpg"
    colors = [get_rand_color(0,255) for x in xrange(max(labels)+1)]

    for label,contour in zip(labels,[x.coords for x in potential_spill_regions]):
        cv2.drawContours(result_image,[contour],-1,colors[label],3)

#    # outline oil in red and false alarms in green
#    cv2.drawContours(result_image,oil,-1,(0,255,0),3)
#    cv2.drawContours(result_image,false_alarms,-1,(0,0,255),3)
#    try:
#        assert len(entropy_list) == len(potential_oil_spills)
#    except:
#        print len(entropy_list),len(potential_oil_spills),"are not equal!"
#    for i,e in enumerate(entropy_list):
#        M = cv2.moments(potential_oil_spills[i])
#        if M['m00'] == 0:
#            print "spill has no area..."
#            print cv2.contourArea(potential_oil_spills[i])
#        else:
#            x,y = (int(M['m10']/M['m00']),int(M['m01']/M['m00']))
#            cv2.putText(result_image,str(round(e,3)),(x,y),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255))

    # save the result image for visual inspection
    cv2.imwrite(result_loc,result_image)

    print "File is:",result_loc

    # return list of contours that were identified as oil
    return None

def show_spill(orig_img,spill,color,width=10):
        cv2.drawContours(orig_img,[spill],0,color,width)
        x,y,w,h =  cv2.boundingRect(spill)
        y-=1
        x-=1
        w+=2
        h+=2
        rect = orig_img[y:y+h,x:x+w]
        print x,y,w,h,rect
        cv2.imshow("Potential Oil Spill",rect)
        cv2.waitKey(0)

def get_rand_color(low,high):
    r,g,b = random.randint(low,high),random.randint(low,high),random.randint(low,high)
    return (r,g,b)

