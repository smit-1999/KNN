import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
def plot(df, test_points):
    plt.scatter(test_points["X1"], test_points["X2"], c=test_points['Y'],s=1)
    plt.scatter(df["X1"], df["X2"], marker="x" )
    plt.xlim(-2.0,2.0)
    plt.ylim(-2.0,2.0)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend
    plt.show()

def generate_test_points():
    return pd.DataFrame([[round(r,2),round(c,2)] for r in np.arange(-2.0,2.1,0.1) for c in np.arange(-2.0,2.1,0.1)])
if __name__ == "__main__":
    df = pd.read_csv('./dataset/D2z.txt', sep=" ",                
                    header=None, names=["X1", "X2", "Y"])    
    test_points = generate_test_points()
    def dist(p1,p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    y_labels = []
    for _index,tp in test_points.iterrows():
        min_dist = 10**6        
        label = '1'
        for index,row in df.iterrows():
            curr_dist = dist([tp[0],tp[1]],[row[0],row[1]])
            min_dist = min(min_dist,curr_dist)
            if min_dist == curr_dist:
                label = row[2]
        y_labels.append(label)
    test_points['Y'] = y_labels
    test_points.columns = ["X1","X2","Y"]
    #print(test_points)
    plot(df, test_points)
