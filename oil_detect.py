#! /usr/bin/python

"""
oil_detect.py
-------
Description: Driver program for oil detection experimentation
Author:      Steve Wilson
Created:     June 13, 2013
"""

import os,argparse,filter_image,time,classify,shutil

def main(image_path,dest_dir,opts={}):
    # load parameters from file
    params = load_params(opts['params_file'])
    letters = ''
    if opts['filters']:
        letters += "_"
        for f in opts['filters']:
            letters += f[0]
    dest_dir.rstrip(os.sep)
    dest_dir += os.sep + time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime()) + letters
    os.makedirs(dest_dir)
    shutil.copy(opts['params_file'],dest_dir)
    # filter the image for dark spots
    if opts['filtered_loc']:
        start_image = opts['filtered_loc']
    else:
        start_image = image_path
    potential_oil_spills = filter_image.apply_filters(image_path,start_image,dest_dir,opts['filters'],params['filter'],opts['visual'],opts['save_tmp'],opts['erase_final'])
    # cleanup dest_dir if it was not used
    if len(os.listdir(dest_dir)) <= 1:
        for f in os.listdir(dest_dir):
            print f
            os.remove(dest_dir+os.sep+f)
        os.removedirs(dest_dir)
    # make a decision about each potential spill
    oil_spills = classify.classify_all(image_path,potential_oil_spills)

def load_params(path):
    contents = ""
    try:
        f = open(path,'r')
        contents = f.read()
        f.close()
    except:
        print("params file could not be opened")
        raise IOError
    return eval(contents)


def parse_cmd_line():
    # handle command line arguments using argparse library
    parser = argparse.ArgumentParser(description="Oil Detect: Automatically locate oil spills in high resolution SAR images.",epilog="NOTE: filters will be applied in the order they are specified")
    parser.add_argument('source_image_path',help="location of image file to be processed")
    parser.add_argument('-d','--dest',help='output destination for processed image dir (default = <current_dir>/processed',dest='dest_dir',default='processed')
    parser.add_argument('-p','--params',help='location of parameters file to use. (default = <current_dir>/params.py',dest='params_file',default='params.py')
    parser.add_argument('-H','--stretch',help='Histogram Stretch - apply during noise reduction phase.',dest='filters',action='append_const',const='histogram_stretch')
    parser.add_argument('-X','--experimental',help='Experimental Filter - apply during noise reduction phase.',dest='filters',action='append_const',const='experimental')
    parser.add_argument('-x','--mean',help='Mean Filter - apply during noise reduction phase.',dest='filters',action='append_const',const='mean')
    parser.add_argument('-m','--median',help='Median Filter - apply during noise reduction phase.',dest='filters',action='append_const',const='median')
    parser.add_argument('-o','--open',help='Opening Filter - apply during noise reduction phase.',dest='filters',action='append_const',const='open_filter')
    parser.add_argument('-c','--close',help='Closing Filter - apply during noise reduction phase.',dest='filters',action='append_const',const='close_filter')
    parser.add_argument('-g','--gaussian',help='Gaussian Blur - apply during noise reduction phase.',dest='filters',action='append_const',const='gaussian')
    parser.add_argument('-s','--sobel',help='Sobel Filter - apply during noise reduction phase.',dest='filters',action='append_const',const='sobel')
    parser.add_argument('-S','--scharr',help='Scharr Filter - apply during noise reduction phase.',dest='filters',action='append_const',const='scharr')
    parser.add_argument('-k','--kmeans',help='K-means - apply during noise reduction phase',dest='filters',action='append_const',const='kmeans')
    parser.add_argument('-l','--lowest',help='Only keep lowest values - apply during noise reduction phase',dest='filters',action='append_const',const='lowest')
    parser.add_argument('-b','--background',help='Remove background values of 0 - apply during noise reduction phase',dest='filters',action='append_const',const='background')
    parser.add_argument('-w','--whites',help='Ceiling filter for bright values during noise reduction phase',dest='filters',action='append_const',const='whites')
    parser.add_argument('-t','--thresh',help='Threshold - apply during noise reduction phase',dest='filters',action='append_const',const='threshold')
    parser.add_argument('-T','--athresh',help='Adaptive Threshold - apply during noise reduction phase',dest='filters',action='append_const',const='adaptive_threshold')
    parser.add_argument('-C','--canny',help='Canny Edge Detection - apply during noise reduction phase',dest='filters',action='append_const',const='canny')
    parser.add_argument('-G','--segmentation',help='Segmentation Algorithm - apply during noise reduction phase',dest='filters',action='append_const',const='segmentation')
    parser.add_argument('-D','--draw',help='draw contours - apply during noise reduction phase',dest='filters',action='append_const',const='draw_contours')
    parser.add_argument('-I','--intermediate',help='save intermediate filtered images',dest='save_tmp',action='store_true')
    parser.add_argument('-E','--erase',help='erase final processed image when done',dest='erase_final',action='store_true')
    parser.add_argument('-V','--vis_all',help='show all steps visually while processing',dest='visual',action='store_true')
    #TODO: implement this: currently both -v and -V will show all steps
    parser.add_argument('-v','--vis',help='show next step visually while processing',dest='visual',action='store_true')
    parser.add_argument('-F','--filtered_img',help='Path to an already filtered image to use',dest='filtered_loc',default=None)
    return parser.parse_args()
    
if __name__ == "__main__":
     args = parse_cmd_line()
     main(args.source_image_path,args.dest_dir,vars(args))
