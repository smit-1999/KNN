import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from collections import defaultdict
from datetime import datetime
dp = defaultdict()
from collections import Counter

def dist(p1,p2):    
    res = 0
    #ignore first column email no, hence start from 1

    #TODO, Recomute dist amtrix from p1[1:n-2]
    updated_p1 = p1[1:]
    updated_p1[updated_p1 == ''] = 0.0
    updated_p1 = updated_p1.astype(float)
    
    updated_p2 = p2[1:]
    updated_p2[updated_p2 == ''] = 0.0
    updated_p2 = updated_p2.astype(float)
    res = np.sqrt(np.sum((updated_p1-updated_p2)**2))

    return res

def get_metrics(y_actual : pd.DataFrame, y_pred: pd.DataFrame) -> float:    
    tp = 0 
    tn = 0
    fp = 0 
    fn = 0
    for i,v in y_actual.items():
        if y_pred[i] == '1': 
            if y_pred[i] == y_actual[i]:
                tp += 1
            else:
                fp += 1
        else:
            if y_pred[i] == y_actual[i]:
                tn += 1
            else:
                fn += 1
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    precision = (tp)/(tp+fn)
    recall = (tp)/(tp+fp)
    return accuracy,precision,recall

def one_nn(df:pd.DataFrame):
    #5 fold validation
    metrics = []
    for i in range(0,5):
        test = df.iloc[(i*1000):1000*(i+1)]
        train = pd.concat([df,test]).drop_duplicates(keep=False)
        y_pred = []
        for x,test_point in test.iterrows():
            min_dist = 10**15       
            label = '1'
            for y,train_point in train.iterrows():
                curr_dist = dp[(x-1,y-1)]
                min_dist = min(min_dist,curr_dist)
                if min_dist == curr_dist:
                    label = train_point['Prediction']
            y_pred.append(label)
        test['Y_pred'] = y_pred
        train_actual = df.iloc[(i*1000):1000*(i+1)]
        y_train = train_actual['Prediction']
        metrics.append(get_metrics(y_train,test['Y_pred']))
    print('Metrics for 5 fold validation:', metrics)
    with open('./dataset/output.txt', 'a') as f: 
        for k in metrics:
            f.write(' '.join([str(i)for i in k]) + " " + str(datetime.now()) + "\n")

def knn(df:pd.DataFrame, k: int)->list:
    metrics = []
    # df=df.head(50)
    for i in range(0,5):
        test = df.iloc[(i*1000):1000*(i+1)]
        train = pd.concat([df,test]).drop_duplicates(keep=False)
        y_pred = []
        print(i,'th fold')
        for x,test_point in test.iterrows():
            label = '1'
            all_distances=[]
            for y,train_point in train.iterrows():
                curr_dist = dp[(x-1,y-1)]
                label = train_point['Prediction']
                all_distances.append((curr_dist,label))
            topK = sorted(all_distances, key = lambda x: x[0])[:k] 
            #print('Top kdistances',topK)
            freq = Counter(elem[1] for elem in topK)
            label = '1' if freq['1'] > freq['0'] else '0'
            #print('freq dict', freq)
            #print('pred label', label)
            y_pred.append(label)
        #print('For i=',i , 'pred:',y_pred)
        test['Y_pred'] = y_pred
        train_actual = df.iloc[(i*1000):1000*(i+1)]
        y_train = train_actual['Prediction']
        metrics.append(get_metrics(y_train,test['Y_pred']))
    print('Metrics for 5 fold validation:', metrics)
    with open('./dataset/output.txt', 'a') as f: 
        for k in metrics:
            f.write(' '.join([str(i)for i in k]) + " " + str(datetime.now()) + "\n")
    return metrics

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
        if i % 100 == 0:
            print('Completed distance calulation for 5points', i-5,'to',i, 'in ', time.time()-prev,'s')
            prev=time.time()
    file = './dataset/distances.txt'
    print('Completed distance calculation for all points in ', time.time()-start,'s')
    with open(file, 'w') as f: 
        for k,v in dp.items():
            f.write(' '.join([str(i)for i in k]) + ' ' + str(v) + "\n")
    
def populate_distance_matrix(file: str)-> None:
    with open(file, 'r') as f: 
        for line in f:
            x , y, val = line.split(" ")
            dp[(int(x),int(y))] = float(val)
    
if __name__ == "__main__":
    df = pd.read_csv('./dataset/emails.csv', sep=",",                
                    header=None, low_memory=False)  
    # set first row in input file as columns of dataframe
    df.rename(columns=df.iloc[0], inplace = True)
    df.drop(df.index[0], inplace = True)
    
    if os.path.exists('./dataset/distances.txt'):
        #if distance matrix exists, store it in dp array
        populate_distance_matrix('./dataset/distances.txt')   
    else:    
        compute_distance_matrix(df)


    #one_nn(df)
    k_metrics = []
    x = [1,3,5,7,10]

    for k in x:
        start = time.time()
        print('Computation began for k :', k)
        metrics = knn(df, k)
        print('Computation for k :', k, 'took', time.time()-start)
        _avg = np.average([i for i,j,k in metrics])
        k_metrics.append(_avg)
        #k_metrics.append(metrics)
        print(k,' ' , metrics)
        

    
    y=k_metrics
    print('X,y',x,y)
    plt.plot(x,y)
    plt.scatter(x,y)
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.legend
    plt.show()

    