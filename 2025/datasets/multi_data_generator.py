import numpy as np
import random
import matplotlib.pyplot as plt
import csv

#Create a dictionary between indices {0,1,2,3,4,5} (for the labels) and elements {{},1,2,12,21,121} of S_3

S3dict = [[],[1],[2],[1,2],[2,1],[1,2,1]]

length = 100
choices = [1,2]

numsamples = 25000

#Function to generate an array of length \"length\" whose elements are sampled from a finite set, \"choices\"
def gensequence(length,choices):
    seq = [random.choice(choices) for _ in range(length)]
    return seq
        
#Function that takes an input sequence \"inputseq\" of 1's and 2's and determines the sequence of elements of S_3 one encounters reading left to right        
def genoutputs(inputseq):
    outputseq= [0] * len(inputseq)
    for i in range(len(inputseq)):
        if i == 0:
            outputseq[i]=inputseq[i]
        elif len(S3dict[outputseq[i-1]])==0:
            outputseq[i]=inputseq[i]
        elif inputseq[i]== S3dict[outputseq[i-1]][-1]:
            outputseq[i]=max(outputseq[i-1]-2,0)
        elif outputseq[i-1] < 5:
            outputseq[i]=min(outputseq[i-1]+2,5)
        else:
            outputseq[i]=outputseq[i-1]-1
    return outputseq

num_train = int(.3*numsamples)
num_test = numsamples - num_train

def quoted_list_str(seq):
    return "[" + ",".join(f"'{x}'" for x in seq) + "]"

#Generate the dataset and write it to two files, otrain.csv and otest.csv
with open('25kmtrain.csv', 'w', newline='') as ftrain, open('25kmtest.csv', 'w', newline='') as ftest:
    writer_train = csv.writer(ftrain)
    writer_test = csv.writer(ftest)
    for i in range(numsamples):
        inputseq = gensequence(length, choices)
        outputseq = genoutputs(inputseq)
        input_str = quoted_list_str(inputseq)
        output_str = quoted_list_str(outputseq)
        writer = writer_train if i < num_train else writer_test
        writer.writerow([input_str, output_str])
    print("All done!")

# python RNN/datasets/
