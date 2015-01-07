
# params.py
# ---------
# Settings file to be loaded and used by the oil detection tool.
# Format is that of nested python dictionaries.
# Can be specified to oil_detect using the -p option.

({
    # ================= #
    # Filter parameters #
    # ================= #
    'filter': ({

        # Mean Operator
        'mean': ({
            'kernel_size_x': 5,
            'max_kernel_size_x': 11,
            'kernel_size_y': 5,
            'max_kernel_size_y': 11,
            # options: rect
            'kernel_shape': 'rect'
        }),

        # Median Operator
        'median': ({
            # function will create a kernel of the following size
            # only 8-bit images can use size > 5
            'kernel_size': 5,
            'max_kernel_size': 5,
            '8bit_kernel_size': 101,
            'whole_image':True
        }),

        # Sobel
        'sobel': ({
            'dx': 1,
            'max_dx': 4,
            'dy': 0,
            'max_dy': 4,
            # use both x and y directions
            # will reverse dx and dy values, 
            #then combine with first result
            'x_and_y': True,
            # weights to use when combining
            # results from first and second direction
            'dir1wgt': .5,
            'dir2wgt': .5
        }) ,

        # Scharr
        'scharr': ({
            'dx': 1,
            'max_dx': 4,
            'dy': 0,
            'max_dy': 4,
            # use both x and y directions
            # will reverse dx and dy values, 
            #then combine with first result
            'x_and_y': True,
            # weights to use when combining
            # results from first and second direction
            'dir1wgt': .5,
            'dir2wgt': .5
        }),

        # Open
        'open_filter': ({
            'kernel_x' : 5,
            'max_kernel_x' : 15,
            'kernel_y' : 5,
            'max_kernel_y' : 15,
            'iterations' : 5,
            'max_iterations': 10
        }),

        # Close
        'close_filter': ({
            'kernel_x' : 5,
            'max_kernel_x' : 15,
            'kernel_y' : 5,
            'max_kernel_y' : 15,
            'iterations' :7,
            'max_iterations' : 10
        }),

        # K-means
        'kmeans': ({
            'whole_image': True,
            'K': 4,
            'max_K':16,
            'max_iter':50,
            'max_max_iter':1000,
            'accuracy':1.0,
            'attempts':1,
            'max_attempts':10,
            'localize': False
        }),

        # Threshold
        'threshold': ({
            'value' : 1000,
            'max_value' : 5000,
            'use_mode':True,
            'whole_image':True
        }),

        # Adaptive Threshold
        'adaptive_threshold': ({
            # Apply to the entire image at once
            # Instead of blocks at a time
            'whole_image': True,
            # Must be odd number
            'blocksize' : 501,
            #'max_blocksize': 150,
            # Constant to be subtracted from mean
            'C': 1,
            'max_C': 100,
            # Factor to apply to mean before thresholding
            'T': .75,
            # compare window mean to global mean when deciding threshold
            'mean_compare':False
        }),

        # White ceiling filter
        'whites': ({
            'whole_image': True
        }),

        # Background removal
        'background': ({
            'whole_image': True
        }),

        # Draw contours
        'draw_contours': ({
            'whole_image': True
        }),

        # Gaussian Blur
        'gaussian': ({
            'kernel_size_x': 5,
            'kernel_size_y': 5,
            'max_kernel_size_x': 51,
            'max_kernel_size_y': 51,
            'sigmaX':      0,
            'max_sigmaX':      10
        }),

        # Canny edge detection
        'canny': ({
            'thresh1': 100,
            'max_thresh1': 255,
            'thresh2': 200,
            'max_thresh2': 255
        }),

        'lowest': ({
            'whole_image':True
        }),

        'segmentation': ({
            'whole_image':False,
#            'k':5000,
            'k':2500,
            'max_k':5000000,
            'no_trackbars':True
        }),

        'experimental': ({
            'whole_image': True,
            'manual_grid': True
        })

    })
})
