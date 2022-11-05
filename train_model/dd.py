import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# import k-folder
from sklearn.model_selection import cross_val_score


if __name__ == '__main__':
    filename = 'test.data'
    data=pd.read_csv(filename,header=None)

    # print(df.head())
    # print(df.isnull().sum())
    print(data[0].value_counts())

    # df[0].value_counts().plot.bar()
    # plt.plot()
    # plt.show()


    # print(x.head())
    # print(y.head())
    # print(x.shape)
    # print(y.shape)

    
    # x = [[0, 1],
    #    [2, 3],
    #    [4, 5],
    #    [6, 7],
    #    [8, 9]]
    # y = [0, 1, 2, 3, 4]

    # z = df.loc[df[0].isin(['H','M'])]
    # print(z)


    # separate result into y and data into x
    df = data.loc[data[0].isin(['H','M'])]
    # df = data.loc[data[0].isin(['M','Y'])]

    x=df.iloc[:,1:]
    y=df.iloc[:,0]

    print(df)
    print(x)
    print(y)
    
    # print(x.head())
    # print(y.head())
    # print(x.shape)
    # print(y.shape)

    # set aside 10% into test variables later
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.1)    

    k_range = range(1, 15)
    k_scores = []
    # use iteration to caclulator different k in models, then return the average accuracy based on the cross validation
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
        k_scores.append(scores.mean())
    # plot to see clearly
    plt.plot(k_range, k_scores)
    plt.xlabel('hyperparameter k')
    plt.ylabel('Cross-Validated Accuracy')
    plt.show()

    '''


    # print(x_train)
    # print(x_test)
    # print(y_train)
    # print(y_test)


    
    # knn = KNeighborsClassifier(algorithm='brute',n_neighbors =1 ,leaf_size=100,p=30)
    knn = KNeighborsClassifier(n_neighbors =2)
    # knn.fit(X_train, y_train)

    # knn_predictions = knn.predict(X_test) 

    # acc=accuracy_score(y_test,knn_predictions)
    # print('Accuracy is :',acc*100)

    # print('accuracy: ', knn.score(X_test, y_test))



    # X,y will automatically devided by 5 folder, the scoring I will still use the accuracy
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    print(scores)
    print(scores.mean())
    


    '''
