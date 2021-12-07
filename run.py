#!/usr/bin/env python

import os
import sys
import json
import pickle
import torch
import numpy as np

sys.path.insert(0, 'src')
# from etl import get_data
from create_test_data import make_test_dir
from model import NeuralNet
from Models import *

def main(targets):
    # if 'data' in targets:
    #     with open('data-params.json') as fh:
    #         data_cfg = json.load(fh)
    #
    #     # make the data target
    #     get_data(**data_cfg)

    if 'test' in targets:

        # if os.path.isdir('./data'):

        #     make_test_dir()

        #     V5 = pickle.load(open('data/V5.p', 'rb'))
        #     active_L_table_slide_DOA = V5["active_L_table_slide_DOA"]


        #     test = open('test/testdata/test_data1.txt', 'w')
        #     output = open('test/output.txt', 'w+')
        #     for i in active_L_table_slide_DOA:
        #         test.write(str(i))

        #     X = torch.from_numpy(active_L_table_slide_DOA[0])
        #     y = torch.from_numpy(np.array([2.0441905444152297, 2.0438931606036195]))

        #     mod = NeuralNet(15, 10, 2)
        #     mod.train(X, y)
        #     predictions = mod.predict(active_L_table_slide_DOA)

        #     for p in predictions:
        #         output.write(str(p))
        
        # default
        data = pickle.load(open('test/testdata/test_data2.p', 'rb'))
        output = open('test/output.p', 'ab')
        table = pickle.load(open('test/testdata/test_data1.p', 'rb'))

        # build x and y train
        cp_torch = [torch.from_numpy(cp) for cp in data[:4]]
        room_coords = [torch.from_numpy(np.array([i[0], i[1]])) for i in ROOM_COORDINATES[:4]]
        X = cp_torch
        y = room_coords

        # model
        mod = NeuralNet(15, 10, 2)
        mod.train(X, y)
        predictions = mod.predict(table)

        # write output
        pickle.dump(predictions, output)
        output.close()

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
