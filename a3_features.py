import os
import sys
import argparse
import numpy as np
import pandas as pd
import glob
from utils import read_text, reduce_dim, label_encoder, shuffle_split
from sklearn import preprocessing
# Whatever other imports you need

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert directories into table.")
    parser.add_argument("inputdir", type=str, help="The root of the author directories.")
    parser.add_argument("outputfile", type=str, help="The name of the output file containing the table of instances.")
    parser.add_argument("dims", type=int, help="The output feature dimensions.")
    parser.add_argument("--test", "-T", dest="testsize", type=int, default="20", help="The percentage (integer) of instances to label as test.")

    args = parser.parse_args()

    print("Reading {}...".format(args.inputdir))

    d = {}
    d1 = []
    X = []
    Y = []
    for infile in glob.glob(args.inputdir + '/*/*/*'):
        dic = {}
        instance = os.path.split(os.path.dirname(infile))[-1]
        review_file = open(infile,'r').read()
        X.append(review_file)
        Y.append(instance)
        if instance not in d:
            d[instance] = []
        d[instance].append(review_file)
    X, _ = read_text(X)
    df = pd.DataFrame(X)
    df = df.fillna(0)
    original_author_names = Y.copy()

    Y =  label_encoder(Y)
    # Do what you need to read the documents here.

    print("Constructing table with {} feature dimensions and {}% test instances...".format(args.dims, args.testsize))
    # Build the table here.
    X = reduce_dim(df, args.dims)

    train_X, test_X, train_Y, test_Y, tag = shuffle_split(X, Y, test_split = args.testsize)
    train_X = pd.DataFrame(train_X)
    test_X = pd.DataFrame(test_X)
    train_Y = pd.DataFrame(train_Y)
    test_Y = pd.DataFrame(test_Y)
    full_dataset_X = pd.concat([train_X, test_X])
    full_dataset_Y = pd.concat([train_Y, test_Y])
    full_dataset_Y = full_dataset_Y.rename(columns = {0 : "labels"})
    combined_X_Y = pd.concat([full_dataset_X, full_dataset_Y], axis = 1)
    combined_X_Y['train_test_tag'] = tag
    combined_X_Y['original_author_name'] = original_author_names


    
    print("Writing to {}...".format(args.outputfile))
    combined_X_Y.to_csv(args.outputfile + ".csv")

    print("Done!")
    
