
"""
filter_image.py
-------
Description: Image filters
Author:      Steve Wilson
Created:     June 13, 2013

Filter functions used for preprocessing.
Run "./oil_detect.py -h" (without quotes) for help.
"""

import os
import shutil
import sys
import math
import random
from itertools import product

import scipy
from scipy import signal
from scipy import cluster
from osgeo import gdal
import numpy as np
import cv2

from sar_image import *

DEBUG = False
BRIGHTEN_FACTOR = 10

# ================
# Filter Functions
# ================
# Do not rename without updating the command line parser in oil_detect.py

# stretch an individual value into a new datasize
# vectorized version is used by histogram stretch
def stretch_value(x,new_max,new_min,old_max,old_min):
    ratio = (new_max-new_min)/(old_max-old_min)
    result = (ratio*(x-old_min))+new_min
    return result

def histogram_stretch(image,params):
    maxval = 2 ** (image.dtype.itemsize * 8)
    minval = 0
    stretch_vector = np.vectorize(stretch_value,otypes=[image.dtype])
    return stretch_vector(image,maxval,minval,params['max'],params['min'])

def mean(image,params):
    if params['kernel_shape'] == 'rect':
        try:
            return cv2.blur(image,(params['kernel_size_x'],params['kernel_size_y']))
        except:
            return cv2.blur(scale_data(image,16),(params['kernel_size_x'],params['kernel_size_y']))
    else:
        sys.stderr.write("kernel shape handler not implemented:"+kernel['shape'])

def median(image,params):
    if image.dtype == np.uint8:
        return cv2.medianBlur(image,params['8bit_kernel_size'])
    else:
        return cv2.medianBlur(scale_data(image,16),params['kernel_size'])

def gaussian(image,params):
    return cv2.GaussianBlur(scale_data(image,16),(params['kernel_size_x'],params['kernel_size_y']),params['sigmaX'])

def sobel(image,params):
    # Sobel(image,ddepth,dx,dy,[dst,ksize,scale,delta,bordertype])
    dir1 = cv2.Sobel(scale_data(image,16),-1,params['dx'],params['dy'])
    if params['x_and_y'] == True:
        dir2 = cv2.Sobel(scale_data(image,16),-1,params['dy'],params['dx'])
        return cv2.addWeighted(dir1,params['dir1wgt'],dir2,params['dir2wgt'],0)
    else:
        return dir1

def scharr(image,params):
    # Scharr(image,ddepth,dx,dy,[dst,scale,delta,bordertype])
    dir1 = cv2.Scharr(scale_data(image,16),-1,params['dx'],params['dy'])
    if params['x_and_y'] == True:
        dir2 = cv2.Scharr(scale_data(image,16),-1,params['dy'],params['dx'])
        return cv2.addWeighted(dir1,params['dir1wgt'],dir2,params['dir2wgt'],0)
    else:
        return dir1

def morphology(image,params,operation):
    scaled_image = scale_data(image,16)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(params['kernel_x'],params['kernel_y']))
    return cv2.morphologyEx(scaled_image,operation,kernel,iterations=params['iterations'])

def open_filter(image,params):
    return morphology(image,params,cv2.MORPH_OPEN)

def close_filter(image,params):
    return morphology(image,params,cv2.MORPH_CLOSE)

def kmeans(image,params):
    
    centroids, labels = cluster.vq.kmeans2(cluster.vq.whiten(np.transpose(image.flatten()[np.newaxis])),params['K'])
    if DEBUG:
        print "centroids"
        print centroids
        print "labels"
        print labels
    return labels.reshape(image.shape)

#    # localize attempts to use pixel distance information as feature
#    if params.get('localize'):
#        values_with_coords = np.empty(image.shape+(3,))
#        for x in xrange(image.shape[0]):
#            for y in xrange(image.shape[1]):
#                values_with_coords[x,y] = np.array([image[x,y],x,y])
#        d3_image = np.float32(values_with_coords).reshape((-1,3))
#        compactness,best_labels,centers = cv2.kmeans(d3_image,params['K'],(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_MAX_ITER,params['max_iter'],params['accuracy']),params['attempts'],cv2.KMEANS_PP_CENTERS)
#        print "...",
#        flat_image = np.copy(image).flatten()
#        print centers
#        for i in xrange(len(flat_image)):
#            flat_image[i] = centers[best_labels[i][0]][0]
#        return flat_image.reshape(image.shape)
#    else:
#        d1_image = np.float32(image).reshape((-1,1))
#        compactness,best_labels,centers = cv2.kmeans(d1_image,params['K'],(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_MAX_ITER,params['max_iter'],params['accuracy']),params['attempts'],cv2.KMEANS_PP_CENTERS)
#        print "...",
#        flat_image = d1_image.flatten()
#        for i in xrange(len(flat_image)):
#            flat_image[i] = centers[best_labels[i][0]]
#        return flat_image.reshape(image.shape)

def threshold(image,params):
    image_u8 = scale_data(image,8)
    if params.get('use_mode'):
        counts = np.bincount(image_u8.flatten())
        val = 1 + np.argmax(counts[1:])
    else:
        val = scale_value(params['value'],8)
    retval,thresh_image = cv2.threshold(image_u8,val,255,cv2.THRESH_BINARY)
    return thresh_image

def adaptive_threshold(image,params):
    if params.get('whole_image'):

        if params.get('mean_compare'):
            image_mean = np.mean(image)
            image_std = np.std(image)
            std_mean_ratio = image_std/image_mean
            b = params['blocksize']
            b -= b % 2
            kernel = np.empty((b,b))
            kernel.fill(1.0/(b*b))
            #kernel[b/2,b/2] = 0
            scaled_image = scale_data(image,16)
            thresh_image = cv2.filter2D(scaled_image,-1,kernel)
            result_image = np.empty(scaled_image.shape,dtype=np.uint8)
            for x in xrange(scaled_image.shape[0]):
                for y in xrange(scaled_image.shape[1]):
                    result_image[x,y] = 255 if scaled_image[x,y] > thresh_image[x,y] * std_mean_ratio else 0
            return result_image
        else:
            b = params['blocksize']
            b -= b % 2
            kernel = np.empty((b,b))
            kernel.fill(1.0/(b*b))
            #kernel[b/2,b/2] = 0
            scaled_image = scale_data(image,16)
            thresh_image = cv2.filter2D(scaled_image,-1,kernel)
            result_image = np.empty(scaled_image.shape,dtype=np.uint8)
            if params['T']:
                thresh_image *= params['T']
            for x in xrange(scaled_image.shape[0]):
                for y in xrange(scaled_image.shape[1]):
                    result_image[x,y] = 255 if scaled_image[x,y] > thresh_image[x,y] else 0
            return result_image
    else:
        image_u8 = scale_data(image,8)
        return cv2.adaptiveThreshold(image_u8,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,params['blocksize'],params['C'])

# throw out outliers- set to black
# should only performed on grayscale images
def whites(image,params):
    xbar = np.mean(image)
    std = np.std(image)
    std3= std*3
    maximum = np.max(image)
    tmp = np.where(image>=std3,0,image)
    return np.where(image==0, maximum, image)

def background(image,orig_image,params):
    filtered_image = np.copy(image)
    for x in xrange(image.shape[0]):
        for y in xrange(image.shape[1]):
            if orig_image[x,y] == 0: filtered_image[x,y] = 255
    return filtered_image

def canny(image,params):
    return cv2.Canny(scale_data(image,8),params['thresh1'],params['thresh2'])

def lowest(image,params):
    image_u8 = scale_data(image,8)
    retval,thresh_image = cv2.threshold(image_u8,np.min(image_u8),255,cv2.THRESH_BINARY)
    return thresh_image

def draw_contours(image,params):
    image_u8 = scale_data(image,8)
    contours,hierarchy = cv2.findContours(image_u8,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    result_image = np.zeros(image_u8.shape,np.uint8)
    result_image[:] = 255
    for c in xrange(len(contours)):
        if cv2.contourArea(contours[c]) > 50:
            cv2.drawContours(result_image,contours,c,0,10)
    return result_image

def experimental(image,output):
    
    def _affinity(x,y,scale):
        return (math.exp(-1*((x.item()-y.item())**2)))/(2*scale**2)

    for block in image:
        all_points = list(product(xrange(block.shape[0]),xrange(block.shape[1])))
        set_points = set()
        total_points = len(all_points)
        sample_size = (block.shape[0]*block.shape[1])/100
        while len(set_points) < sample_size:
            set_points.add((random.choice(all_points)))
        other_points = list(set(all_points) - set_points)
        sample = list(set_points)
        assert len(sample) == sample_size
        A = np.empty((sample_size,sample_size),dtype=np.float64)
        B = np.empty((sample_size,len(other_points)),dtype=np.float64)
        for s1 in xrange(sample_size):
            for s2 in xrange(sample_size):
                if s1==s2:
                    A[s1,s2] = 0
                else:
                    A[s1,s2] = _affinity(block[sample[s1]],block[sample[s2]],.1)
            for other in xrange(len(other_points)):
                B[s1,other] = _affinity(block[sample[s1]],block[other_points[other]],.1)
        if DEUBG:
            print "A",A.shape
    #        print A
            print "B",B.shape
    #        print B
        d1 = sum(np.vstack((A,B.T)),0)
        d2 = sum(B,0) + np.dot(np.dot(sum(B.T,0),np.linalg.pinv(A)),B)
        if DEBUG:
            print "d1,d2",d1.shape,d2.shape
    #        print d1
    #        print d2
        d1d2 = np.hstack((d1,d2))
        dhat = np.transpose(np.where(d1d2!=0,(1.0/d1d2)**.5,0)[np.newaxis])
        if DEBUG:
    #        print "dhat",dhat.shape
            print dhat
        dhat[np.isinf(dhat)] = 0
        A = A * np.dot(dhat[:sample_size],np.transpose(dhat[:sample_size]))
#        print B.shape,dhat[:sample_size].shape,dhat[sample_size:sample_size+total_points].shape
        B = B * np.dot(dhat[:sample_size],np.transpose(dhat[sample_size:sample_size+total_points]))
#        B[np.isinf(B)] = 0
        if DEBUG:
            print "ASHAPE:",A.shape
        asi = scipy.linalg.sqrtm(np.linalg.pinv(A))
#        asi = np.where(A!=0,A**-.5,0)
        asi[np.isinf(asi)] = 0
        if DEBUG:
            print "asi",asi.shape
    #        print asi
        Q = A+np.dot(np.dot(np.dot(asi,B),np.transpose(B)),asi)
        if DEBUG:
            print "Q",Q.shape
    #        print Q
        U,L0,T = np.linalg.svd(Q)
        L = np.zeros(Q.shape,dtype=np.float64)
        L[:T.shape[0],:T.shape[0]] = np.diag(L0)
        if DEBUG:
            print "U L T"
            print U
            print L
            print T
        V = np.dot(np.dot(np.dot(np.vstack((A,np.transpose(B))),asi),U),np.linalg.pinv(np.sqrt(L)))
        E = None
        if DEBUG:
            print "V",V.shape
            print V
        for ii in xrange(1,3):
            vec = V[:,ii]/V[:,0]
            if DEBUG:
                print "vec",vec.shape
                print vec
            if E==None:
                E = vec
#                print "E",E.shape
#                print E
            else:
                E = np.vstack((E,vec))
#                print "E",E.shape
#                print E
        centroid,label = cluster.vq.kmeans2(E.T,2)
        if DEBUG:
            print centroid,label
        result = np.empty(block.shape)
        for ii in xrange(sample_size):
            result[sample[ii]] = label[ii]
        for jj in xrange(len(other_points)):
            result[other_points[jj]] = label[sample_size+jj]
#        print result
        cv2.namedWindow("block",cv2.CV_WINDOW_AUTOSIZE)
        cv2.namedWindow("result",cv2.CV_WINDOW_AUTOSIZE)
        cv2.imshow("block",block)
        cv2.imshow("result",result)
        cv2.waitKey(0)

#        affinity_matrix = np.empty((num_points,num_points))
#        degree_matrix = np.zeros((num_points,num_points))
#        for x in xrange(num_points):
#            row_sum = 0
#            for y in xrange(len(all_points)):
#        #        print block[points[x]],block[points[y]]
#                val = 0
#                if x != y:
#                    val = _affinity(block[points[x]],block[points[y]],.1)
#                row_sum += val
#                affinity_matrix[x,y] = val
#            degree_matrix[x,x] = row_sum
#
#        print affinity_matrix
#        print degree_matrix
#        print degree_matrix**-.5
#        D = degree_matrix**-.5
#        # set all inf values to 0
#        # these were created by doing 0**.5
#        D[np.isinf(D)] = 0
#        L = np.dot(D,np.dot(affinity_matrix,D))
#        print "L",L
#        eigvals,eigvects = np.linalg.eig(L)
#        print "points",points
#        print ""
#        print "eigvects",eigvects
#        EV1 = eigvects[0]
#        EV2 = eigvects[1]
#        X = np.dstack((EV1,EV2))[0]
#        row_sums = X.sum(axis=1)
#        print X
#        print row_sums
#        Y = X/row_sums[:,np.newaxis]
#        print Y
#        centroid,label = cluster.vq.kmeans2(Y,2)
#        print centroid,label
#        result = np.empty(block.shape)
#        for ii in xrange(len(points)):
#            result[points[ii]] = label[ii]
#        print result

#def experimental(image,params):
#    if params.get('whole_image'):
#        line_image = np.copy(image)
#        # find lines in x direction
#        diffs = []
#        prev = None
#        xlines = []
#        ylines = []
#        for x in xrange(image.shape[0]):
#            total = sum(image[x,y] for y in xrange(image.shape[1]))
#            if prev:
#                diffs.append([abs(total-prev),x])
#            prev = total
#        print "x direction:",sorted([diff[0] for diff in diffs],reverse=True)[:20]
#        a = np.array([diff[0] for diff in diffs])
#        m = a.mean()
#        s = a.std()
#        t = m+s
#        print m, s, t
#        for d,x in diffs:
#            if d > t:
#               print 'drew a line: x',x
#               cv2.line(line_image,(0,x),(image.shape[1],x),0,10) 
#        diffs = []
#        prev = None
#        for y in xrange(image.shape[1]):
#            total = sum(image[x,y] for x in xrange(image.shape[0]))
#            if prev:
#                diffs.append([abs(total-prev),y])
#            prev = total
#        print "y direction:",sorted([diff[0] for diff in diffs],reverse=True)[:20]
#        a = np.array([diff[0] for diff in diffs])
#        m = a.mean()
#        s = a.std()
#        t = m+s
#        print m, s, t
#        for d,y in diffs:
#            if d > t:
#               print 'drew a line: y',y
#               cv2.line(line_image,(y,0),(y,image.shape[0]),0,10) 
#        return line_image
#    else:
#        test = np.copy(image)
#        cv2.line(test,(0,0),(500,500),255,10)
#        return test

def segmentation(image,params):
   
#    image = src_image[1024:1024+512,7168:7168+512]

    class Component:
        
        i = 0

        def __init__(self, vertices=[], Int=0, ID=None, size=1,):
            self.Int = Int
            if not ID:
                self.ID = Component.i
                Component.i += 1
            else:
                self.ID = ID
            self.size = size
            self.vertices = vertices
            self.color = -1

    class Segmentation:

        def __init__(self,V):
            self.components = {}
            for v in V:
                C = Component([v])
                self.components[v] = C

        def merge(self,v1,v2,C1,C2,w):
            for v in C2.vertices:
                self.components[v] = C1
            C1.vertices.extend(C2.vertices)
            C2.vertices = None
            C1.Int = w
            C1.size += C2.size
            C2 = None

#        def get_component(self,v):
#            for i,C in enumerate(self.components):
#                if v in C.vertices:
#                    return C,i
#            print "could not find a component for:",v

    def _MInt(C1,C2):
        return min( C1.Int + params['k']/C1.size, C2.Int + params['k']/C2.size )

    def _find_weight(v1,v2):
        return abs(int(image[v1])-int(image[v2]))

    def _get_neighbors(v,h,w):
        y,x = v
        locs = [-1,0,1]
        N = []
        for dx in locs:
            for dy in locs:
                if 0<=dx+x<w and 0<=dy+y<h:
                    N.append((dy+y,dx+x))
        return N
   
    if DEBUG:
        print ""
        print "building graph..."
    Component.i = 0
    height,width = image.shape
    V = list(product(xrange(image.shape[0]),xrange(image.shape[1])))
    E = {}
    for v in V:
        for n in _get_neighbors(v,height,width):
            i = tuple(sorted([v,n]))
            if i not in E:
                w = _find_weight(v,n)
                E[i] = w

    if DEBUG:
        print "Running segmentation..."
        print "Step 0"
    # Step 0
    pi = sorted(E.iterkeys(),key = lambda o:E[o])
    if DEBUG:
        print "Step 1"
    # Step 1
    S = Segmentation(V)
    if DEBUG:
        print "Step 2"
    # Step 2
        print "Step 3"
    # Step 3
    for o in pi:
#        print o
        v1,v2 = o
        w = E[o]
        C1,C2 = S.components[v1],S.components[v2]
#        print "checking for merge: ",C1.ID,C2.ID,w,_MInt(C1,C2)
        if C1.ID != C2.ID:
            if w<=_MInt(C1,C2):
#                print "performed merge!"
                # Bigger component should be listed first for faster processing
                if len(C1.vertices) > len(C2.vertices):
                    S.merge(v1,v2,C1,C2,w)
                else:
                    S.merge(v2,v1,C2,C1,w)
#            else:
#                print "didn't merge because w=",w,"and MINT=",_MInt(C1,C2)
#        print "didn't merge becasue",C1.ID,"=",C2.ID
#        print "segmentations:",len(S.components)

    if DEBUG:
        print "...done"
    result = np.zeros(image.shape)
    if DEBUG:
        print "counting segmentations..."
        print "segmentations:",len(set(S.components.values()))
        print "creating result image..."
    for C in set(S.components.values()):
#        total = 0
        mini = float('infinity')
        for v in C.vertices:
#            total += image[v]
            if image[v] < mini:
                mini = image[v]
#        C.color = int(total/len(C.vertices))
        C.color = mini
    for v in V:
        C = S.components[v]
        result[v] = C.color

    return result

#    print "running kmeans..."
#    centroids, labels = cluster.vq.kmeans2(cluster.vq.whiten(np.transpose(result.flatten()[np.newaxis])),10)
#    labels2d = labels.reshape(image.shape)
#    lowest = np.min(centroids)
#    kmeans_image = np.empty(image.shape,dtype=np.uint8)
#    for x in xrange(image.shape[0]):
#        for y in xrange(image.shape[1]):
#            if centroids[labels2d[x,y]] <= lowest:
#                kmeans_image[x,y] = 0
#            else:
#                kmeans_image[x,y] = 255
#
#    cv2.namedWindow("orig")
#    cv2.namedWindow("clusters")
#    cv2.namedWindow("kmeans")
#    cv2.imshow("orig",scale_data(image,8)*10)
#    cv2.imshow("clusters",scale_data(result,8)*10)
#    cv2.imshow("kmeans",kmeans_image)
#    cv2.waitKey(0)
#    sys.exit()
#
#    return kmeans_image

# ================
# Helper Functions
# ================

def get_window(arr,x,y,length,width):
    ldiv,lmod = divmod(length,2)
    wdiv,wmod = divmod(width,2)
    sy = max(y-ldiv,0)
    sy2= y + ldiv + lmod
    sx = max(x-wdiv,0)
    sx2= x + wdiv + wmod
    return arr[sy:sy2,sx:sx2]

def scale_data(arr,depth):
    arr2 = np.copy(arr)
    while arr2.max() > 2**depth - 1:
        arr2 = np.sqrt(arr2)
    return np.array(arr2,dtype=eval('np.uint'+str(depth)))
    
def scale_value(val,depth):
    while val > 2**depth -1:
        val = math.sqrt(val)
    return val
       
# ================
# Driver Functions
# ================

# returns a unique file path for the given path and extension
# appends a number to the end of the filename
def get_unique_name(path,ext):
    orig_path = path
    i = 0
    while os.path.exists(path+ext):
        i += 1
        path = orig_path + '_' + str(i)
    return path + ext

def show_with_trackbar(window_name,p,filter_function,image,params):
    if p in params and "max_"+str(p) in params and type(params[p]) == type(int()):
        if DEBUG:
            print "adding trackbar"
        def callback(x):
            params[p] = x
            filtered_image = filter_function(image,params)
            cv2.imshow(window_name,filtered_image)
        cv2.createTrackbar(p,window_name,params[p],params['max_'+p],callback)
        callback(params[p])
    else:
        if DEBUG:
            print
            print "Can't use trackbar..."
            print p
            print params
        filtered_image = filter_function(image,params)
        if DEBUG:
            print filtered_image
        cv2.imshow(window_name,scale_data(filtered_image,8))
#        cv2.imshow(window_name,filtered_image)

# calls each individual filter function
def run_filter(filter_function,sar_image,dest,ext,params,vis):

    # check metadata to see if create/copy allowed for file type
    metadata = sar_image.driver.GetMetadata()
    filtered_image = None

    # if copy allowed, make a copy
    if metadata.has_key(gdal.DCAP_CREATECOPY) and metadata[gdal.DCAP_CREATECOPY] == 'YES':
        filtered_image = sar_image.driver.CreateCopy(get_unique_name(dest,ext), sar_image.image, 0)
        filtered_band = filtered_image.GetRasterBand(1)

    # otherwise, make a GTiff file
    else:

        # load GTiff driver and change file extension to .tiff
        if DEBUG:
            sys.stderr.write("gdal cannot copy the type of image being used\n")
            sys.stderr.write("converting to gtiff...\n")
        gtiff_driver = gdal.GetDriverByName("GTiff")
        ext = '.tiff'

        # Parameters for Create() shown below
        # Create(file_path,xdim,ydim,num_bands,datatype)
        filtered_image = gtiff_driver.Create(get_unique_name(dest,ext),sar_image.image.RasterXSize,sar_image.image.RasterYSize,1,gdal.GDT_UInt32)
        filtered_band = filtered_image.GetRasterBand(1)

    # Setup windows for visualization
    if vis and not params.get('whole_image'):
        orig_window_name = "Original"
        filter_window_name = "Filter: " + filter_function.__name__
        cv2.namedWindow(orig_window_name,cv2.CV_WINDOW_AUTOSIZE)
        cv2.namedWindow(filter_window_name,cv2.CV_WINDOW_AUTOSIZE)
        cv2.moveWindow(filter_window_name,sar_image.x_blocksize,0)
        def trackbar_changed(pos):
            params['%s'] = pos
            filtered_block = filter_function(block,params)
            imshow(filter_window_name,filtered_block*BRIGHTEN_FACTOR)

    # Get max/min here and add to params if function name = histogram_stretch
    if filter_function.__name__ == "histogram_stretch":
        band = sar_image.image.GetRasterBand(1)
        params['min'],params['max'] = band.ComputeStatistics(1)[0:2]

    # check if operation is set to be applied to whole image instead of blocks at a time
    if params.get('whole_image') or params.get('manual_grid'):
        if params.get('manual_grid'):
            filter_function(sar_image,filtered_band)
        else:
            # process entire image as a whole
            matrix = sar_image.to_array()
            filtered_matrix = filter_function(matrix,params)
            filtered_band.WriteArray(filtered_matrix)
    else:
        # process each block of the image
        for block in sar_image:

            # run the filter function on one block
            filtered_block = filter_function(block,params)

            # display the block
            if vis:
                sys.stderr.write(str(block))
                try:
                    cv2.imshow(orig_window_name,block*BRIGHTEN_FACTOR)
                except:
                    cv2.imshow(orig_window_name,scale_data(block,16))
                if params.get("no_trackbars"):
                    try:
                        cv2.imshow(filter_window_name,filtered_block*BRIGHTEN_FACTOR)
                    except:
                        cv2.imshow(filter_window_name,scale_data(filtered_block,16))
                else:
                    for p in params.keys():
                        show_with_trackbar(filter_window_name,p,filter_function,block,params)
                cv2.waitKey()

            # write processed band back to gdal image object
            filtered_band.WriteArray(filtered_block,sar_image.prev_x,sar_image.prev_y)

    # Get the filename of the filtered image
    filtered_image_location = filtered_image.GetDescription()

    # Store the process step in the image's metadata
    past_ops = ''
    metadata = filtered_image.GetMetadata()
    if 'process' in metadata:
        past_ops = metadata['process'] + ', '
    filtered_image.SetMetadataItem('process',past_ops + filter_function.__name__)

    # this will cause gdal to flush and write image to file
    filtered_image = None
    return filtered_image_location

# Sequentially applys the filters in filter_list to the image
# save_tmp keeps a copy of the image at each step
def apply_filters(original,src,dest,filter_list,params,visual=False,save_tmp=False,erase_final=False):
    print "Running Filters..."
    tmp = src

    # get the name and extension of the original file
    orig_name = src.split(os.sep)[-1]
    orig_no_ext, ext = orig_name.split('.',1)
    ext = '.' + ext

    # iterate through the filters
    if filter_list:
        for index,f in enumerate(filter_list):
            print '\t',f,"...",
            sys.stdout.flush()

            # create an empty dict if no params loaded
            if f not in params: params[f] = {}

            # create a Sar_Image object
            image = Sar_Image(tmp)

            # get a reference to the actual function object
            func = eval(f)

            # run the filter specified by f
            file_dest = dest+os.sep+str(index)+'_'+f
            filtered_loc = run_filter(func,image,file_dest,ext,params[f],visual)

            # must do this so gdal knows to close the input image
            image.image = None
            print "\t\t...done"

            # remove old tmp file if unwanted
            if not save_tmp and tmp != src:
                os.remove(tmp)
            tmp = filtered_loc
            ext = '.' + tmp.split('.',1)[-1]

    print "Processing Complete"
    print "Extracting contours...",
    sys.stdout.flush()
    potential_oil = get_contours(original,tmp)
    print "\t...done"

    # save with original extension if file type was not changed
    # otherwise ext will have been changed to .tiff
    final = get_unique_name(dest+os.sep+orig_no_ext + "_filtered",ext)
    if filter_list and tmp!=src:
        if save_tmp:
            shutil.copy(tmp,final)
        else:
            shutil.move(tmp,final)
        if erase_final:
            os.remove(final)
            print "No filtered image file was saved."
        else:
            print "Filtered image file is:" + final
    return potential_oil

def get_contours(orig,src):
    sys.stdout.flush()
    orig_image = Sar_Image(orig)
    orig_matrix = orig_image.to_array()
    orig_u8 = scale_data(orig_matrix,8)
    retval,mask = cv2.threshold(orig_u8,0,255,cv2.THRESH_BINARY_INV)
#    cv2.imwrite("mask.jpg",mask)
#    print(set(mask.flatten()))
    dilated_mask = cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)),iterations=2)
#    cv2.imwrite("dil_mask.jpg",dilated_mask)
#    print(set(dilated_mask.flatten()))
    sar_image = Sar_Image(src)
    matrix = sar_image.to_array()
    image_u8 = scale_data(matrix,8)
#    cv2.imwrite("image.jpg",image_u8)
#    print(set(image_u8.flatten()))
    # Ensure that black border does not get selected as a contour
    image_u8 += dilated_mask
    whites = image_u8 > 0
    image_u8[whites] = 255
#    cv2.imwrite("imagewithmask.jpg",image_u8)
#    print(set(image_u8.flatten()))
    contours,hierarchy = cv2.findContours(255-image_u8,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    sar_image.image = None
    orig_image.image = None
    return contours
