
"""
sar_image.py
-------
Description: Class to deal with Sar Images
Author:      Steve Wilson
Created:     June 25, 2013
"""

from osgeo import gdal
import sys

# blocksize to use when iterating through sar image via for loop
X_BLOCK = 512
Y_BLOCK = 512

# Class to deal with SAR Imagery more effectively
# Allows iteration through blocks using a for loop
class Sar_Image:

    x_blocksize = X_BLOCK
    y_blocksize = Y_BLOCK

    def __init__(self,path):
        self.image = gdal.Open(path)
        self.driver = self.image.GetDriver()
        self.current_x = 0
        self.current_y = 0
        self.prev_x = None
        self.prev_y = 0

    def __iter__(self):
        return self

    # return the next block from the image
    # blocksize is specified by the class variables:
    #    x_blocksize, y_blocksize
    def next(self):
        x_block = self.x_blocksize
        y_block = self.y_blocksize
        # right edge has been reached/passed
        if self.current_x >= self.image.RasterXSize:
            self.prev_x = self.current_x
            self.current_x = 0
            self.prev_y = self.current_y
            self.current_y += y_block
        # bottom edge has been reached/passed
        if self.current_y >= self.image.RasterYSize:
            raise StopIteration
        # almost at right edge, use smaller width for block
        if self.current_x + self.x_blocksize > self.image.RasterXSize:
            x_block = self.image.RasterXSize - self.current_x
        # almost at bottom edge, use smaller height for block
        if self.current_y + self.y_blocksize > self.image.RasterYSize:
            y_block = self.image.RasterYSize - self.current_y
        # deal with multiple bands here - currently not implemented
        #if self.image.RasterCount > 1:
        #    sys.stderr.write("WARNING: Input image has multiple bands. Multiple band processesing is not yet implemented")
        # Get array data from current block
        band = self.image.GetRasterBand(1)
        image_array = band.ReadAsArray(self.current_x,self.current_y,x_block,y_block,x_block,y_block)
        self.prev_x = self.current_x
        self.prev_y = self.current_y
        self.current_x += x_block
        return image_array

    def to_array(self,x=0,y=0,width=None,height=None,bandno=1):
        band = self.image.GetRasterBand(bandno)
        return band.ReadAsArray(x,y,width,height)

    def get_dimensions(self):
        return (self.image.RasterXSize,self.image.RasterYSize)
