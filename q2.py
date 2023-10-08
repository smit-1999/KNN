import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import json,pickle
import time
import os
from collections import defaultdict
dp = defaultdict()

def dist(p1,p2):    
    res = 0
    #ignore first column email no, hence start from 1
    updated_p1 = p1[1:]#np.delete(p1,0)
    updated_p1[updated_p1 == ''] = 0.0
    updated_p1 = updated_p1.astype(float)
    
    updated_p2 = p2[1:]#np.delete(p2,0)
    updated_p2[updated_p2 == ''] = 0.0
    updated_p2 = updated_p2.astype(float)
    res = np.sqrt(np.sum((updated_p1-updated_p2)**2))

    return res

def get_accuracy(train : pd.DataFrame, test: pd.DataFrame) -> float:    
    tp = 0 
    tn = 0
    fp = 0 
    fn = 0
    y_pred = test['Y_pred']
    y_actual = train['Prediction']
    for i in range(0,len(y_pred)):
        if y_pred[i] == 1: 
            if y_pred[i] == y_actual[i]:
                tp += 1
            else:
                fp += 1
        else:
            if y_pred[i] == y_actual[i]:
                tn += 1
            else:
                fp += 1
    return (tp+tn)/(tp+tn+fp+fn)
def one_nn(df):
    test = df.iloc[0:1000]
    train = df.iloc[1000:5001] 
    #print(test,train)
    y_pred = []
    for _index,test_point in test.iterrows():
        min_dist = 10**6        
        label = '1'
        for index,train_point in train.iterrows():
            curr_dist = dist(train_point, test_point)
            min_dist = min(min_dist,curr_dist)
            if min_dist == curr_dist:
                label = train_point['Prediction']
        y_pred.append(label)
    test['Y_pred'] = y_pred
    acc = get_accuracy(train,test)

def compute_distance_matrix(df: pd.DataFrame) -> None:
    prev=time.time()
    start=time.time()
    for i in range(0, len(df)):
        for j in range(i+1, len(df)):
            p1 = df.iloc[i].to_numpy()
            p2 = df.iloc[j].to_numpy()
            distance = dist(p1,p2)
            dp[(i,j)] = distance if (i,j) not in dp.keys() else distance
            dp[(j,i)] = distance if (j,i) not in dp.keys() else distance
        #print('Completed distance calulation for ', i, 'in ', end-start,'s')
        if i % 100 == 0:
            print('Completed distance calulation for 5points', i-5,'to',i, 'in ', time.time()-prev,'s')
            prev=time.time()
    file = './dataset/distances.txt'
    print('Completed distance calculation for all points in ', time.time()-start,'s')
    with open(file, 'w') as f: 
        for k,v in dp.items():
            f.write(' '.join([str(i)for i in k]) + ' ' + str(v) + "\n")
    

if __name__ == "__main__":
    df = pd.read_csv('./dataset/emails.csv', sep=",",                
                    header=None, low_memory=False)  
    df.rename(columns=df.iloc[0], inplace = True)
    df.drop(df.index[0], inplace = True)
    if os.path.exists('./dataset/distances.txt'):
        with open('./dataset/distances.txt', 'r') as f: 
            for line in f:
                x , y, val = line.split(" ")
                dp[(int(x),int(y))] = float(val)
        print(dp[(0,1)],dp[(1,0)]) 
    else:    
        compute_distance_matrix(df)
    
    