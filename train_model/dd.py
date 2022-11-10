import sys
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
from sklearn.svm import SVC
# for dimension reduction
from sklearn.feature_selection import SequentialFeatureSelector


# for order by desc. return second element for sort
def takeAccu(elem):
    return elem[1]
    

# analyze on the input data set using several models 
# data: dataframe of input file
# type: kNN, RandomForest, DecisionTree, SVM
# problem: what to classfy
# dimension_reduction: 'Y' yes, 'No' no
def analyze(data, model_type, problem, dimension_reduction = 'N'):
    

    for p in problem:
        # separate data into paramters(x) and a value(y)
        df = data.loc[data[0].isin(p)]
        x=df.iloc[:,1:]
        y=df.iloc[:,0]
        print('Classification problem: {}'.format(p))
        print('\tPicked data: x {}, y {}\n'.format(x.shape, y.shape))

        # for test
        if False:
            x=df.iloc[:100,1:]
            y=df.iloc[:100,0]

        ###################################
        # dimension reduction
        ###################################
        if dimension_reduction == 'Y':
            x = greedyBackward(model_type, x, y)

        ###################################
        # set aside 10% for a test later
        ###################################
        X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.1)   

        ###################################
        # set category hyperparameters for each model
        ###################################
        if model_type == 'kNN':
            categories = ['ball_tree','kd_tree','brute']
        elif model_type == 'RandomForest':
            categories = ['gini','entropy','log_loss']
        elif model_type == 'DecisionTree':
            categories = ['gini','entropy','log_loss']
        elif model_type == 'SVM':
            categories = ['rbf']


        for category in categories:
            k_scores = []
            hyper_score = []
            k_range = range(1, 11)
            if model_type == 'SVM':
                k_range = [0.001, 0.01, 0.02, 0.03, 0.05, 0.1, 0.5, 1, 5, 10]

            ###################################
            # cross validation to find proper hyper parameter
            ###################################            
            for k in k_range:
                if model_type == 'kNN':
                    model = KNeighborsClassifier(algorithm=category, n_neighbors=k)
                elif model_type == 'RandomForest':
                    model = RandomForestClassifier(criterion=category, max_depth=k)
                elif model_type == 'DecisionTree':
                    model = DecisionTreeClassifier(criterion=category, max_depth=k)
                elif model_type == 'SVM':
                    model = SVC(kernel=category, C=k)

                # do a Cross-validation test
                scores = cross_val_score(model, X_train, y_train, cv=5)
                k_scores.append(scores.mean())
                if True:
                    print('Testing: cross-vali, model: {}, category haper: {}, number hyper: {}, score: {}'.format(model_type, category, k, scores.mean()))

                hyper_score.append((k, scores.mean()))

            ###################################
            # choose best hyperparameter from the hyper value we test above
            # pick the hyper parameter that yeilds the hightest score(or accuracy) 
            hyper_score.sort(key=takeAccu, reverse=True)
            bestHyper = hyper_score[0][0]

            ###################################
            # test with a final validation set
            ###################################
            start = time.time()
            if model_type == 'kNN':
                model = KNeighborsClassifier(algorithm=category, n_neighbors=bestHyper)
            elif model_type == 'RandomForest':
                model = RandomForestClassifier(criterion=category, max_depth=bestHyper)
            elif model_type == 'DecisionTree':
                model = DecisionTreeClassifier(criterion=category, max_depth=bestHyper)            
            elif model_type == 'SVM':
                model = SVC(kernel=category, C=bestHyper)

            model.fit(X_train, y_train)            
            finalScore = model.score(X_test, y_test)
            end = time.time()
            print('Result : final-vali, model: {}, category hyper: {}, number hyper: {}, score: {}, time: {gap:.4f}\n'.format(model_type, category, bestHyper, finalScore, gap = (end - start)))

            ###################################
            # show plot to see clearly
            ###################################
            if True:
                plt.plot(k_range, k_scores)                    
                plt.title('{} {} - Reduced: {}, Categorical hyper: {}'.format(p, model_type, dimension_reduction, category))
                plt.ylabel('Cross-Validated Accuracy')
                if model_type == 'kNN':
                    plt.xlabel('Numerical hyper: k(how many neighbors)')
                elif model_type == 'RandomForest':
                    plt.xlabel('Numerical hyper: k(max depth)')
                elif model_type == 'DecisionTree':
                    plt.xlabel('Numerical hyper: k(max depth)')
                elif model_type == 'SVM':
                    plt.xlabel('Numerical hyper: k(C, Regularization parameter)')
                plt.show()

# reduce dimension on the input x, y
# model_type: 'kNN', 'RandomForest', 'DecisionTree', 'SVM'
# x,y: input data
# howmany: how many features do you want to left
def greedyBackward(model_type, x, y, howmany = 4):
    print('Greedy Backward is in progress.')
    print('\tbefore x shape: {}'.format(x.shape))

    if model_type == 'kNN':
        model = KNeighborsClassifier()
    elif model_type == 'RandomForest':
        model = RandomForestClassifier(max_depth=8)
    elif model_type == 'DecisionTree':
        model = DecisionTreeClassifier(max_depth=8)
    elif model_type == 'SVM':
        model = SVC()

    start = time.time()
    sfs = SequentialFeatureSelector(model, direction='backward', n_features_to_select=howmany, cv = 5)
    reduced_x = sfs.fit_transform(x,y)
    end = time.time()

    print('\tafter x shape: {}'.format(reduced_x.shape))
    print('Reduced x: model {}, left features: {}, time: {gap:.4f}\n'.format(model_type, sfs.get_feature_names_out(), gap = (end - start)))

    return reduced_x

def usage(exec_name):
    print('Usage:')
    print('\tpython %s [arg1] [arg2] [arg3] [arg4]' % (exec_name))
    print('\targ1: input data file')
    print('\targ2: choose model. "kNN" "RandomForest" "DecisionTree" "SVM" are available')
    print('\targ3: choose problem. 1:H,K 2:M,Y 3:I,J 4:H,K,M,Y,I,J') 
    print('\targ4: apply dimension reduction. Y,y:yes N,n:no') 
    print('\t\tGreedy Backward Feature Elimination will be applied')
    print('Example:')
    print('\tpython %s test.data kNN 1 N' % (exec_name))
    print('\tpython %s test.data RandomForest 2 Y' % (exec_name))
    print('\tpython %s test.data DecisionTree 3 N' % (exec_name))
    print('\tpython %s test.data SVM 4 Y' % (exec_name))
    exit()

# check arguments and return arguments
def checkarg(argv):
    if len(argv) != 5:
        usage(argv[0])
    else :
        model_type = argv[2]
        problem = argv[3]
        reduction = argv[4].upper()

        if not (argv[2] == 'kNN' or argv[2] == 'RandomForest' or argv[2] == 'DecisionTree' or argv[2] == 'SVM'):
            print('Error:')
            print('\tcheck the argument [%s]' % (argv[2]))
            usage(argv[0])

        if not (argv[3] == '1' or argv[3] == '2' or argv[3] == '3' or argv[3] == '4'):
            print('Error:')
            print('\tcheck the argument [%s]' % (argv[3]))
            usage(argv[0])

        if not (argv[4] == 'Y' or argv[4] == 'N'):
            print('Error:')
            print('\tcheck the argument [%s]' % (argv[4]))
            usage(argv[0])
        
        test_problem = []
        if problem == '1':
            test_problem = [['H','K']]
        elif problem == '2':
            test_problem = [['M','Y']]
        elif problem == '3':
            test_problem = [['I','J']]
        elif problem == '4':
            test_problem = [['H','K','M','Y','I','J']]

        return [model_type, test_problem, reduction]


if __name__ == '__main__':

    # check program parameter
    ret_arg = checkarg(sys.argv)
    model_type = ret_arg[0]
    problem = ret_arg[1]
    reduction = ret_arg[2]

    data=pd.read_csv(sys.argv[1],header=None)
    print('Loading data from file [{}]'.format(sys.argv[1]))
    print('\t Loaded data: {}\n'.format(data.shape))
    
    analyze(data, model_type, problem, reduction)





def reduce_analyze(data, model_type, problem):
    for p in problem:
        print('Classification problem: {}'.format(p))

        # separate data into paramters(x) and a value(y)
        df = data.loc[data[0].isin(p)]

        x=df.iloc[:,1:]
        y=df.iloc[:,0]
        print('x shape: {}'.format(x.shape))
        print('y shape: {}'.format(y.shape))

        model = RandomForestClassifier(max_depth=8)
        model.fit(x,y)
        print('reduction1', model.score(x,y))

        sfs = SequentialFeatureSelector(model, direction='backward', n_features_to_select=4, cv = 5)
        z = sfs.fit_transform(x,y)

        print('x shape: {}'.format(x.shape))
        print('y shape: {}'.format(y.shape))
        print('x {}'.format(x))
        print('z {}'.format(z))
        print('z {}'.format(sfs.get_feature_names_out()))


        # # print('after', sfs.she)
        model2 = RandomForestClassifier(max_depth=8)
        model2.fit(z,y)
        print('reduction2', model2.score(z,y))

# my implement Greedy Backward Selection
def myGreedyBack(data):
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





