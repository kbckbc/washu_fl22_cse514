import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# for test data set
from sklearn.model_selection import train_test_split
# for cross-validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
# for classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier



# for order by desc. return second element for sort
def takeAccu(elem):
    return elem[1]
    
# data: dataframe of input file
# type: kNN, RandomForest, SVM
def training(data, model_type):

    problem = [['H','K'], ['M','Y'], ['I','J']]
    # problem = [, ['O','Q']]
    for p in problem:
        print('Classify problem: {}'.format(p))

        # separate data into paramters(x) and a value(y)
        df = data.loc[data[0].isin(p)]
        x=df.iloc[:,1:]
        y=df.iloc[:,0]

        if False: 
            print('df\n{}'.format(df))
            print('x shape: {}'.format(x.shape))
            print('x\n{}'.format(x))
            print('y shape: {}'.format(y.shape))
            print('y\n{}'.format(y))

        # set aside 10% for a test later
        X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.1)   

        # for kNN
        # use 3 category hyperparameters, those are 'ball_tree','kd_tree','brute'
        # use 10 number hyperparamters, 1 ~ 10
        if model_type == 'kNN':
            categories = ['ball_tree','kd_tree','brute']
        elif model_type == 'RandomForest':
            categories = ['gini','entropy','log_loss']
        elif model_type == 'DecisionTree':
            categories = ['gini','entropy','log_loss']

        for category in categories:
            k_scores = []
            hyper_score = []
            k_range = range(1, 11)

            # use iteration to caclulator different k in models, then return the average accuracy based on the cross validation
            for k in k_range:
                if model_type == 'kNN':
                    model = KNeighborsClassifier(algorithm=category, n_neighbors=k)
                elif model_type == 'RandomForest':
                    model = RandomForestClassifier(criterion=category, max_depth=k)
                elif model_type == 'DecisionTree':
                    model = DecisionTreeClassifier(criterion=category, max_depth=k)


                # 
                # do a Cross-validation test
                scores = cross_val_score(model, X_train, y_train, cv=5)
                k_scores.append(scores.mean())
                if True:
                    print('Testing: cross-vali, model: {}, category: {}, test hyper: {}, score: {}'.format(model_type, category, k, scores.mean()))

                hyper_score.append((k, scores.mean()))

            # plot to see clearly
            if False:
                plt.plot(k_range, k_scores)
                plt.xlabel('hyperparameter k (model:{} category: {})'.format(model_type, category))
                plt.ylabel('Cross-Validated Accuracy')
                plt.show()

            # choose best hyperparameter from the hyper value we test above
            # pick the hyper parameter that yeilds the hightest score(or accuracy) 
            hyper_score.sort(key=takeAccu, reverse=True)
            bestHyper = hyper_score[0][0]

            # test with a final validation set
            start = time.time()
            if model_type == 'kNN':
                model = KNeighborsClassifier(algorithm=category, n_neighbors=bestHyper)
            elif model_type == 'RandomForest':
                model = RandomForestClassifier(criterion=category, max_depth=bestHyper)
            elif model_type == 'DecisionTree':
                model = DecisionTreeClassifier(criterion=category, max_depth=bestHyper)            
            model.fit(X_train, y_train)
            end = time.time()
            print('Result : final-vali, model: {}, category: {}, final hyper: {}, score: {}, time: {gap:.4f}\n'.format(model_type, category, bestHyper, model.score(X_test, y_test), gap = (end - start)))





def reduction(data):
    # separate data into paramters(x) and a value(y)
    # df = data.loc[data[0].isin(['A'])]
    df = data
    x=df.iloc[:,1:]
    y=df.iloc[:,0]

    # print(x.columns)

    # X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.1)   


    # model = KNeighborsClassifier(n_neighbors=1)
    # score = cross_val_score(model, X_train, y_train, cv=5)
    # print('score {}'.format(score))

    if False: 
        print('df\n{}'.format(df))
        print('x shape: {}'.format(x.shape))
        print('x\n{}'.format(x))
        print('y shape: {}'.format(y.shape))
        print('y\n{}'.format(y))

    while len(x.columns) > 4:
        print(len(x.columns))

        # model = KNeighborsClassifier(n_neighbors=3)
        # score = cross_val_score(model, x, y, cv=5).mean()

        scores = []
        for i in x.columns:
            xx = x.copy()
            xx.pop(i)
            # print('xx {}'.format(xx))
            model = KNeighborsClassifier(n_neighbors=1)
            score = cross_val_score(model, xx, y, cv=5)
            print('cross-valdiation: w/o i {}, score {}'.format(i, score.mean()))
            scores.append((i, score.mean()))    

        scores.sort(key=takeAccu, reverse=True)
        # print('aa:{}'.format(scores))
        pick = scores[0][0]
        x.pop(pick)

    print('x {}'.format(x))

        
    # xx = df.iloc[:,1:]
    # xx.pop(1)
    # print('xx\n{}'.format(xx))
    # print('xx\n{}'.format(x))
    # X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.1)   
    # model = KNeighborsClassifier(n_neighbors=1)
    # model.fit(X_train, y_train)
    # print(model.score(X_test, y_test))



if __name__ == '__main__':
    debug = True
    filename = 'test.data'
    data=pd.read_csv(filename,header=None)

    
    # training(data, 'kNN')
    # training(data, 'RandomForest')
    training(data, 'DecisionTree')

    # reduction(data)







    '''
    k_range = range(1, 11)
    k_scores = []
    # use iteration to caclulator different k in models, then return the average accuracy based on the cross validation
    for k in k_range:
        model = KNeighborsClassifier(n_neighbors=k)

        fold = 0
        skf = StratifiedKFold(n_splits=5)
        fold_scores = []
        for train_index, test_index in skf.split(X_train, y_train):
            # print("TRAIN:", len(train_index), "TEST:", len(test_index))
            train_x = X_train.iloc[train_index,:]
            train_y = y_train.iloc[train_index]

            test_x = X_train.iloc[test_index,:]
            test_y = y_train.iloc[test_index]

            model.fit(train_x, train_y)
            fold_score = model.score(test_x, test_y)
            fold_scores.append(fold_score)

            if False:
                print('fold: {}, accuracy: {}'.format(fold, fold_score))

            fold += 1

        k_score = model.score(X_test, y_test)
        k_scores.append(k_score)
        print('k: {}, cross accuracy: {}, test data accuracy: {}'.format(k, np.array(fold_scores).mean(), k_score))
        # print('k: {}, test data accuracy: {}'.format(k, model.score(X_test, y_test)))

    plt.plot(k_range, k_scores)
    plt.xlabel('hyperparameter k')
    plt.ylabel('Cross-Validated Accuracy')
    plt.show()

    '''



    '''


    print('a')

    # print(df.head())
    # print(df.isnull().sum())
    # print(data[0].value_counts())

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
    # df = data.loc[data[0].isin(['H','M'])]
    # 
    #     


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
