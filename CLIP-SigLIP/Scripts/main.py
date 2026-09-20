import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re,nltk,json
from bs4 import BeautifulSoup
### ML Librarires--------------------
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from sklearn.metrics import average_precision_score,roc_auc_score, roc_curve, precision_recall_curve
from imblearn.metrics import macro_averaged_mean_absolute_error
###-------------------------------------------
np.random.seed(42)
import random
import warnings
warnings.filterwarnings('ignore')

import sys
import argparse
import time
import dataset as d
import models as m
import evaluation as e


'''Evaluation Parameters'''

def print_metrices(true,pred):
    # print(confusion_matrix(true,pred))
    print(classification_report(true,pred,target_names=['NoAg','GAg','PAg','RAg','Oth'],digits = 3))
    print("Precison : ",precision_score(true,pred, average = 'weighted'))
    print("Recall : ",recall_score(true,pred,  average = 'weighted'))
    print("F1 : ",f1_score(true,pred,  average = 'weighted'))
    print("Macro F1 : ",f1_score(true,pred,  average = 'macro'))
    print("MMAE: ",macro_averaged_mean_absolute_error(true, pred))



def main(args):
    start_time = time.time()
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming the root directory is one levels up from where the script is located
    root_dir = os.path.abspath(os.path.join(script_dir, ".."))
    # Construct the path to the dataset folder relative to the script directory
    dataset_base_path = os.path.join(root_dir, args.dataset_path)
    # Construct the full path by appending the 'Files' folder and the Excel file
    excel_path = os.path.join(dataset_base_path, '')
    memes_path = os.path.join(dataset_base_path, 'Img')
    # create a path model saving
    saved_models_dir = os.path.join(root_dir, args.model_path)
    # Create the folder if it doesn't already exist
    os.makedirs(saved_models_dir, exist_ok=True)


    ## Load the Processed Data Splits
    train_loader, valid_loader, test_loader = d.load_dataset(excel_path,
                                                              memes_path,
                                                              args.maximum_length,
                                                              args.batch)
                                                        
    #train the model 
    start_time = time.time()
    actual, pred = e.pipline(train_loader, 
                              valid_loader, 
                              test_loader, 
                              saved_models_dir,
                              args.n_heads,
                              args.epochs, 
                              args.lr_rate)

    # #  evaluation
    print(f"Classification Report :")
    print_metrices(actual, pred)

    end_time = time.time()
    print(f"Total time :{end_time-start_time:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bengali Hateful Memes Classification')

    parser.add_argument('--dataset', dest='dataset_path', type=str, default = 'Dataset',
                        help='the directory of the dataset folder')
    parser.add_argument('--max_len', dest='maximum_length', type=int, default = 70,
                        help='the maximum text length - default 70')
    parser.add_argument('--batch_size',dest="batch", type=int, default = 16,
                        help='Batch Size - default 16')   
    parser.add_argument('--model', dest='model_path', type=str, default = 'Saved_Models',
                        help='the directory of the saved model folder')
    parser.add_argument('--heads',dest="n_heads", type=int, default = 16,
                        help='number of heads - default 16')                       
    parser.add_argument('--n_iter',dest="epochs", type=int, default = 5,
                        help='Number of Epochs - default 1')
    parser.add_argument('--lrate',dest="lr_rate", type=float, default = 5e-5,
                        help='Learning rate - default 5e-5')
                     

    
    args = parser.parse_args()
    main(args)