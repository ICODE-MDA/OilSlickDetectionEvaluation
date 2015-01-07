        ==============
         Experimental
        Oil  Detection
            System
        ==============
-------------------------------
        Steve Wilson
        NREIP Intern
      SPAWAR - Pacific
      June-August 2013
-------------------------------
===========
Description
===========
python code to expriment with oil detection, including:
    - preprocssing image filters
    - contour detection
    - feature extraction
    - classification framework*
        *under construction, needs classifier development

a small oil slick dataset, including:
    - 20 image chips of oil slicks
    - 20 image chips of lookalikes
    - relevant metadata

===========
Quick Start
===========

The top level program is oil_detect.py, which can be run with:

    ./oil_detect.py <path_to_SAR_product> [-flags[-opt <args>[-opt2 <args>[...]]]]

where path_to_SAR_product must be a file type that GDAL can open
(see www.gdal.org/formats_list.html).

For a full list of options and flags that can be used, do:

    ./oil_detect.py -h

Many of the flags correspond to image filters to be run. 
As described in the help message, these will be applied in the order that they are listed.
For example:

    ./oil_detect.py -Hxm my_product.xml

will first apply histogram stretching, then a mean filter, and finally a median filter

    -----------------
    Choosing Filters:
    -----------------
    Each time the system is run, the filters must be specified with command line arguments.
    While experimentation is may lead to even better results, the following sequence was found to be fairly effective:
        -adaptive threshold
        -median
        -segmenation
        -open
        -close

=================
Filter Parameters
=================

The file params.py will be loaded by default. The format is simply that of a python dictionary.
NOTE: before changing parameters in the file, it may be useful to make a backup of the original params.py:

    cp params.py backup_params.py

To use a different file as the parameters file, use the -p option:

    ./oil_dectect.py -p my_params_file.py my_product.xml

NOTE2: some functions require parameters to exist! ommitting some parameters in your file may cause errors.

    -------------------
    Special parameters:
    -------------------

    1. SLIDER MAX
    There will often be a my_param and a my_param_max for a given filter
    The my_param_max specifies the max value to be used when visualizing the results with a slider.
    The use of a slider requires specifying a max value.
    NOTE: This does NOT mean that you cannot manually set the parameter my_param to a value larger than my_param_max...
        you just cannot do this with the slider during visualization

    2. WHOLE IMAGE
    A common parameter is called whole_image. 
    This means that the filter will be applied globally instead of being applied to one block at a time.
    It is required because normally, only one block is passed to the filter at a time. The code that calls
        the filter functions needs to know to pass the entire image instead.
    NOTE: This often causes an explicit call to a differnt block of code than the original filter.

=======
Results
=======

By default, results will be saved in the directory ./processed/<year_month_date_hour_min_sec_filterlist>/
This can be specified by using the -d option:

    ./oil_detect.py -d my_results_dir my_product.xml

To save the results after EACH filter, use the intermediate results flag, -i:

    ./oil_detect.py -ixxx my_product.xml

will run a mean filter 3 times, saving the succesive outputs to a file.
NOTE: results will be save in the .gtiff format if GDAL does not have writing capabilities for the file type that was opened.

=========
Debugging
=========

To visualize the results of a filter, use the -v command:

    ./oil_detect.py -xv my_product.xml

The -x applies a mean filter, and the -v shows the results of this filter for each block.
This means that the user must manually press a key to cycle through blocks, which will become impractical for large images.

print statements are sprinkled throughout the files. There may be a global variable DEBUG at the top of a file.
Setting DEBUG = True should turn on some additional output.

=====
Files
=====
Below is a list of the files used and a brief description of each.

&&&&&&&&&&&&
python_code/
&&&&&&&&&&&&

-------------
oil_detect.py
-------------
Driver program for oil detection.
    -loads parameters
    -creates results dir
    -runs the following:
        --filter_image.py
        --classify.py

---------------
filter_image.py
---------------
A basic image processing suite.
    -Allows for filters to be run on image:
        --mean
        --median
        --gaussian
        --histogram stretch
        --graph-based-segmentation
        --adaptive threshold
        --... (do "./oil_detect.py -h" for more)
    -Returns all contours detected on final processed image
NOTE: many filters will automatically cause downscaling of the image
    (i.e. bit-depth will be reduced when certain filters are run)

-----------
classify.py
-----------
classification framework
    -creates chips for each contour
    -generates feature vectors for each chip
    -passes feature vectors to a classifier
    -draws a result image with:
        --red outlines around instances of oil slick
        --green outlines around false alarms

------------
sar_image.py
------------
sar_image class
    -handles opening and closing of SAR product files
    -allows iteration through image via for loops
    -handles i/o of raster products
NOTE: set BLOCK_SIZE global params at the top of this file to change the block size used for processing.

----------------
feature_funcs.py
----------------
feature extraction functions
    -loaded into a list that can be iterated through
    -any function with the @use_function decorator will be used to create a feature dictionary during classification

&&&&&&&&&&&&&&&&&&&&&&&&&&&
oil_classification_dataset/
&&&&&&&&&&&&&&&&&&&&&&&&&&&

------------------
vectorize_chips.py
------------------
chip processing tool
    -creates features.txt for each chip containing
        --list of all features and their calculated values
        --coordinates of contours for the dark region in the image
    -generates a sample image of the region
        --shows the detected contour

------
ann.py
------
Artificial Neural Network trainer
    -finds the features.txt files created by vectorize_chips
    -trains a netural network based on the dataset
    -gives output about classifier accuracy using leave-one-out

==============
Remaining Work
==============
Doing a:

    "grep -i -n TODO *.py"

should provide a list of unfinished business that was ommitted during development due to lack of time

