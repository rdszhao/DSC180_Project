import os
import sys
import numpy as np
import pickle

# Load picke data and create data file
def make_test_dir():
    test_directory = './test'
    test_data_directory = './test/testdata'

    # Create test directory and its subdirectory
    if not os.path.exists(test_data_directory):
        # Create parent dir
        if not os.path.exists(test_data_directory):
            os.mkdir(test_directory)
        os.mkdir(test_data_directory)


