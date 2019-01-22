from __future__ import print_function

import sys
import time
import os

import numpy as np

from numba import jit, stencil

from PIL import Image

@stencil()
def gaussian_blur_3d(a):
    return (a[-2,-2,0] * 0.003  + a[-1,-2,0] * 0.0133 + a[0,-2,0] * 0.0219 + a[1,-2,0] * 0.0133 + a[2,-2,0] * 0.0030 +
            a[-2,-1,0] * 0.0133 + a[-1,-1,0] * 0.0596 + a[0,-1,0] * 0.0983 + a[1,-1,0] * 0.0596 + a[2,-1,0] * 0.0133 +
            a[-2, 0,0] * 0.0219 + a[-1, 0,0] * 0.0983 + a[0, 0,0] * 0.1621 + a[1, 0,0] * 0.0983 + a[2, 0,0] * 0.0219 +
            a[-2, 1,0] * 0.0133 + a[-1, 1,0] * 0.0596 + a[0, 1,0] * 0.0983 + a[1, 1,0] * 0.0596 + a[2, 1,0] * 0.0133 +
            a[-2, 2,0] * 0.003  + a[-1, 2,0] * 0.0133 + a[0, 2,0] * 0.0219 + a[1, 2,0] * 0.0133 + a[2, 2,0] * 0.0030)


def main (*args):
    iterations = 10
    
    if len(args) > 0:
        input_file = args[0]
    else:
        raise ValueError("A jpeg file must be provided as the first command line parameter.")

    if len(args) > 1:
        iterations = int(args[1])

    parts = os.path.splitext(input_file)
    new_file_name = parts[0] + "-blur" + parts[1]

    input_img = Image.open(input_file)
    input_arr = np.array(input_img)

    tstart = time.time()
    for i in range(iterations):
        output_arr = gaussian_blur_3d(input_arr)
        input_arr, output_arr = output_arr, input_arr
    htime = time.time() - tstart
    print("SELFTIMED ", htime)

    output_arr = input_arr.astype(np.uint8)

    new_img = Image.fromarray(output_arr, mode=input_img.mode)
    new_img.format = input_img.format
    new_img.save(new_file_name)
    input_img.close()

    input_img = Image.open(input_file)
    input_arr = np.array(input_img)

    #tstart = time.time()
    #output_arr = input_arr.copy()
    #gaussian_blur_std_3d(input_arr, output_arr)
    #input_arr, output_arr = output_arr, input_arr

    #htime = time.time() - tstart
    #print("Standard Python time", htime)
    input_img.close()

if __name__ == "__main__":
    main(*sys.argv[1:])



