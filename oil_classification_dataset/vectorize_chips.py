#! /usr/bin/python

"""
vectorize_chips.py

Steve Wilson
August 2013

Will create feature.txt files with all features activated in feature_funcs.py
Can be run from command line on top level directory:
    python vectorize_chips.py oilSlickClassification/chips/
"""

import os
import sys
import xml.etree.cElementTree as ET

from osgeo import gdal
from itertools import product
import numpy as np
import cv2

import feature_funcs
import ann

# Used for segmentation method
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

# Used for segmentation method
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

# Class for a Region of Interest
# Each chip will be represented as one
class Region:

    # path is original file location
    def __init__(self,path,classification=None):
        self.path = path
        self.features = {}
        self.classification = classification
        # try to have gdal open the file
        try:
            image = gdal.Open(path)
            assert image != None
        except:
            # in the case of the .dim files, gdal won't work correctly
            # so, we try to help point it in the right direction
            if path.split(os.sep)[-1].split('.')[1] == "dim":
                try:
                    # special loader for the .dim chip files
                    image = self.load_DIMAP(path)
                except:
                    # still didn't work...
                    raise IOError("Error reading DIMAP file:", + path)
            else:
                # GDAL couldn't open, and it was not a .dim file
                print "not DIMAP:",path.split(os.sep)[-1].split('.')[1]
                raise IOError("GDAL unable to open " + path)
        # TODO add support for polarimetric data by reading multiple bands here
        band = image.GetRasterBand(1)
        self.mat = band.ReadAsArray()
        self.filtered = [self.mat]
        self.f_labels = ['original']
        self.mask = None
        self.inside_points = None
        self.contours = []

    # function to help gdal find the image files for DIMAP format metadata
    def load_DIMAP(self,path):
        print "Using custom file opening process..."
        tree = ET.parse(path)
        root = tree.getroot()
        datafiles = root.findall("./Data_Access/Data_File/DATA_FILE_PATH")
        for df in datafiles:
            header_path = df.attrib['href']
            print "found header file:",header_path
            image = gdal.Open( os.sep.join(path.split(os.sep)[:-1]) + os.sep + header_path.rsplit('.',1)[0] + '.img' )
            print "success!"
            return image

    # active contour approach
    def snake(self):

        # scale the result to fit 0-255 range
        result = self.mat[:]
        result -= np.min(result)
        result *= (255.0/(np.max(result)-np.min(result)))
        assert np.max(result) <= 255
        assert np.min(result) >= 0
        image_u8 = np.empty(result.shape,dtype='uint8')
        np.copyto(image_u8,result,'unsafe')

        # initialize the snake to a circle
        w,h = self.mat.shape
        circle_image = np.zeros((w,h),np.uint8)
        cv2.circle(circle_image,(h/2,w/2),w/3,255)
        snake = []
        circle_points = np.transpose(np.where(circle_image==255))

        # a different way to initialize the snake:
#        contours,_ = cv2.findContours(circle_image,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
#        for c in contours:
#            for outer in c:
#                for point in outer:
#                    snake.append(tuple(point))
#        snake = []
#        for p in circle_points:
#            print p
#            snake.append(tuple(p))

        # snake evolution parameters
        a = [.5]
        b = [.3]
        c = [.1]

        i = 0
        # old opencv (cv2.cv) functions need a special format, so do some conversion here
        bitmap = cv2.cv.CreateImageHeader((self.mat.shape[1],self.mat.shape[0]),cv2.cv.IPL_DEPTH_8U,1)
        cv2.cv.SetData(bitmap,image_u8.tostring(),image_u8.dtype.itemsize * 1 * self.mat.shape[1])
        # iterate manually so results can be displayed
        while i < 200:
            # create a mask image
            mask = cv2.cv.CloneImage(bitmap)
            # update the snake
            snake = cv2.cv.SnakeImage(mask,snake,a,b,c,(19,19),(cv2.cv.CV_TERMCRIT_ITER,1,0.001))
            # copy the image to draw the snake on it
            copyimage = image_u8.copy()

            # use this to draw just the points
            #for point in snake:
            #    print point
            #    cv2.circle(copyimage,point,5,255,-1)

            # this will draw connected points
            cv2.drawContours(copyimage,[np.array(snake)],0,255,1)
            cv2.imshow("snaking",copyimage)
            cv2.waitKey(10)

            # increment iteration
            i += 1

    def denoise(self,kernel_size=5):
        median_image = cv2.medianBlur(self.filtered[-1],kernel_size)
        self.filtered.append(median_image)
        self.f_labels.append("blur")

    """
    implementation of:
    Felzenszwalb, Pedro F., and Daniel P. Huttenlocher. "Efficient graph-based image segmentation." International Journal of Computer Vision 59.2 (2004): 167-181.

    see paper for algorithm details.
    """
    def segment(self,k=2500):

        def _MInt(C1,C2):
            return min( C1.Int + k/C1.size, C2.Int + k/C2.size )

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

        image = self.filtered[-1]
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
        # Step 0
        pi = sorted(E.iterkeys(),key = lambda o:E[o])
        # Step 1
        S = Segmentation(V)
        # Step 2
        # Step 3
        for o in pi:
            v1,v2 = o
            w = E[o]
            C1,C2 = S.components[v1],S.components[v2]
            if C1.ID != C2.ID:
                if w<=_MInt(C1,C2):
                    # Bigger component should be listed first for faster processing
                    if len(C1.vertices) > len(C2.vertices):
                        S.merge(v1,v2,C1,C2,w)
                    else:
                        S.merge(v2,v1,C2,C1,w)
        result = np.zeros(image.shape)
        for C in set(S.components.values()):
            mini = float('infinity')
            for v in C.vertices:
                if self.mat[v] < mini:
                    mini = self.mat[v]
            C.color = mini
        for v in V:
            C = S.components[v]
            result[v] = C.color

        # scale to 8 bit here:
        result -= np.min(result)
        result *= (255.0/(np.max(result)-np.min(result)))
        assert np.max(result) <= 255.1
        assert np.min(result) >= 0
        image_u8 = np.empty(result.shape,dtype='uint8')
        np.copyto(image_u8,result,'unsafe')
        self.filtered.append(image_u8)
        self.f_labels.append("segmentation")
        # use this to see how many segments were produced
        #print "segments:",len(set(S.components.values()))

    def scale_to_8bit(image):
        # make sure original is not changed
        result = image.copy()
        result -= np.min(result)
        result *= (255.0/(np.max(result)-np.min(result)))
        assert np.max(result) <= 255.1
        assert np.min(result) >= 0
        image_u8 = np.empty(result.shape,dtype='uint8')
        # normally unsafe, but we already checked bounds above
        np.copyto(image_u8,result,'unsafe')
        return image_u8

    # set threshold based on 2 sigma below mean of entire region of interest
    def threshold(self,blocksize=25,C=5):
        
        # scale to 8 bit here
        result = self.filtered[-1].copy()
        result -= np.min(result)
        result *= (255.0/(np.max(result)-np.min(result)))
        assert np.max(result) <= 255.1
        assert np.min(result) >= 0
        image_u8 = np.empty(result.shape,dtype='uint8')
        np.copyto(image_u8,result,'unsafe')
        avg = np.mean(image_u8)
        std = np.std(image_u8)
        result,img = cv2.threshold(image_u8,int(avg-std*2),255,cv2.THRESH_BINARY)
        self.filtered.append(img)
        self.f_labels.append("threshold")

    """
    Options for op include:
        1) MORPH_OPEN
        2) MORPH_CLOSE
        3) MORPH_GRADIENT
        4) MORPH_TOPHAT
        5) MORPH_BLACKHAT
    """
    def morph(self,op,x_size=3,y_size=3,iters=1):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(x_size,y_size))
        result = cv2.morphologyEx(self.filtered[-1],op,kernel,iterations=iters)
        self.filtered.append(result)
        self.f_labels.append("morph")

    # create a mask of area inside contours
    # and save the pixel locations of the contours
    def mask_contours(self):
        image_u8 = self.filtered[-1]
        contours,hierarchy = cv2.findContours(255-image_u8,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        self.mask = np.zeros(self.mat.shape,np.uint8)
        cv2.drawContours(self.mask,contours,-1,255,-1)
        self.inside_points = np.nonzero(self.mask)
        self.contours = [c for c in contours if cv2.contourArea(c)>50]

    # create a feature dictionary for each region
    def calc_features(self):
        all_features = {name:func(self) for name,func in feature_funcs.feature_dict()}
        # this only retains features that were actually decorated
        # all_features will also contain any features used for calculations
        self.features = {k:v for (k,v) in all_features.items() if k in feature_funcs.labels()}

    # visualize all steps
    def show_filter_process(self):
        for i in range(len(self.filtered)):
            cv2.imshow(self.f_labels[i],self.filtered[i]*10)
            cv2.waitKey(1000)

    # write feature dictionary to file
    # and save a preview of the image with contour drawn
    def write(self):

        # build the output paths
        parent_path,filename = self.path.rsplit(os.sep,1)
        name,ext = filename.split('.',1)
        if ext == 'dim':
            parent_path += os.sep + name + '.data'
        featuresfile = parent_path + os.sep + "features.txt"
        imagefile = parent_path + os.sep + "preview.png"

        # create the features file here
        f = open(featuresfile,'w')
        for k,v in self.features.items():
            f.write(str(k)+' : '+str(v)+'\r\n')
        for i,c in enumerate(self.contours):
            f.write("contour_"+str(i)+":\r\n")
            for shell in c:
                for point in shell:
                    f.write('\t'+str(point[0])+","+str(point[1])+"\r\n")
        f.close()

        # changing to 8bit here:
        result = self.mat.copy()
        result -= np.min(result)
        result *= (255.0/(np.max(result)-np.min(result)))
        assert np.max(result) <= 255.1
        assert np.min(result) >= 0
        image_u8 = np.empty(result.shape,dtype='uint8')
        np.copyto(image_u8,result,'unsafe')

        # draw contours and save image file
        cv2.drawContours(image_u8,self.contours,-1,255,1)
        cv2.imwrite(imagefile,image_u8)
        
# apply this process to each chip
# some things are commented out - they may be useful for experimentation
# but the method here works fairly well
def process_chip(chip_path,classification):
    roi = Region(chip_path,classification)
    roi.denoise(kernel_size=5)
    roi.threshold(blocksize=101,C=0)
    #roi.segment(k=2500)
    #roi.snake()
    #roi.threshold(blocksize=101,C=0)
    roi.morph(cv2.MORPH_OPEN,x_size=3,y_size=3,iters=1)
    #roi.morph(cv2.MORPH_CLOSE,x_size=3,y_size=3,iters=1)
    roi.mask_contours()
    roi.calc_features()
    #roi.show_filter_process()
    roi.write()

# find all image chips and process them
def process_all(top_level_dir):
    if not os.path.isdir(top_level_dir):
        usage()
    else:
        # we are assuming the hierarchy is:
        # top_level_dir/class_dir/chip_file
        for class_dir in os.listdir(top_level_dir):
            # make sure class dir is actually a dir
            if os.path.isdir(top_level_dir + os.sep + class_dir):
                for chip_file in os.listdir(top_level_dir + os.sep + class_dir):
                    # make sure chip_file is actually a file
                    if not os.path.isdir(top_level_dir + os.sep + class_dir + os.sep + chip_file):
                        print "processing file:",chip_file
                        process_chip(top_level_dir + os.sep + class_dir + os.sep + chip_file, class_dir)

def usage():
    print "usage: python vectorize_chips.py TOP_LEVEL_DIR"

if __name__ == "__main__":
    args = sys.argv
    if len(args) != 2:
        usage()
    else:
        process_all(args[1].rstrip(os.sep))
