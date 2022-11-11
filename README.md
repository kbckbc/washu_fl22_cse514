# washu_fl22_cse514
22 Fall semester CSE 523(Data mining) course works at Washington University in St. Louis.


# Programming Assignment #1
[Open requirements](https://github.com/kbckbc/washu_fl22_cse514/blob/main/linear_regression/Programming%20Assignment%201.pdf)


+ This project is to find a linear function on some data sets using Linear regression methodology and gradient descent algorithm for optimization. Linear regression is a statistical technique that models the linear relationship between a dependent variable y and one or more independent variables x. The gradient descent refers to changes in the model moving along the slope or slope of the graph toward the lowest possible error value. 
+ The data set is about the strength of concrete when concrete is mixed up with several ingredients. In the data set, 8 feature values affect the strength of concrete. 
+ More specifically, I’m going to build a program to figure out the proper linear model by adopting Uni-variate and Multi-variate regression. Uni-variate linear regression is y = mx + b and multi-variate is y = b0 + m1*x1 + m2*x2 + .. + mn*xn. And I will use MSE(Mean Squared Error) as a loss function. The program can use one feature from the data set or all 8 features to find a linear function. It can be done by designating parameters.

## How to run
```

python gg.py
Usage:
        python gg.py [arg1] [arg2] [arg3] [arg4] [arg5]
        arg1: input excel file
        arg2: train or test
                train - apply learned model to a training data set
                test - apply learned model to a test data set
        arg3: mse or mae
                mse - using mse as a loss function
                mae - using mae as a loss function
        arg4: nopre or prezero or premean
                nopre - no pre-processing on the data
                prezero - pre-processing data by deleting zero values
                premean - pre-processing data by substituing zero with mean value
        arg5: plot or noplot
                plot - show a plot on a uni-varite model
                noplot - not showing a plot on a uni-varite model
Example:
        python gg.py Concrete_Data.xls train mse nopre noplot
        python gg.py Concrete_Data.xls train mse nopre plot
        python gg.py Concrete_Data.xls train mae nopre noplot
        python gg.py Concrete_Data.xls train mae nopre plot
        python gg.py Concrete_Data.xls test mse nopre noplot
        python gg.py Concrete_Data.xls test mse nopre plot
        python gg.py Concrete_Data.xls test mae nopre noplot
        python gg.py Concrete_Data.xls test mae nopre plot
```

## Some images of the result

![pic](https://github.com/kbckbc/washu_fl22_cse514/blob/main/linear_regression/cement.png)
![pic](https://github.com/kbckbc/washu_fl22_cse514/blob/main/linear_regression/coarse.png)

## Lesson learned
+ First column, Cement, was the best compared to other columns in accuracy. And the third one, Fly Ash, was the worst compared to the other columns. Because Cement data points had a tendency like a linear line, it seemed it had no outlier data. On the other hand, Fly ash seemed to have a lot of outliers (which was zero value). It was hard to find a proper linear line with such data. I think that the more data that is out of tendency, the less accurate it is.

+ Same models can predict the same result on the testing data. I think it shows us the power of Linear regression method.

+ From the test I’ve been doing so far, the top 4 factors are useful to predict Concrete compressive strength. Cement, Age, Superplasticizer, Water. They usually had higher VarianceExplain(r square) values than the others. A higher r square value can show that the factor has a significant impact on the linear model.
+ It is judged that the optimum strength is to be shown by putting more than 500 kg of cement, about 150 to 160 kg of water, and 10 to 13 kg of superplasticizer, and then hardening it for 50 to 100 days.


# Programming Assignment #2
[Open requirements](https://github.com/kbckbc/washu_fl22_cse514/blob/main/train_model/Programming%20Assignment%202.pdf)

+ In this project, I will use several models to classify alphabet images into actual characters and verify whether similar results can be obtained through dimensionality reduction. I will evaluate which model is better based on the alphabet recognition performance and time taken. If this project is successful, it can be used for automatic text conversion of images, and furthermore, it can be used for general-purpose character recognition programs.

## How to run
```
Usage:
        python dd.py [arg1] [arg2] [arg3] [arg4]
        arg1: input data file
        arg2: choose model. "kNN" "RandomForest" "DecisionTree" "SVM" are available
        arg3: choose problem. 1:H,K 2:M,Y 3:I,J 4:H,K,M,Y,I,J
        arg4: apply dimension reduction. Y,y:yes N,n:no
                Greedy Backward Feature Elimination will be applied
Example:
        python dd.py test.data kNN 1 N
        python dd.py test.data RandomForest 2 Y
        python dd.py test.data DecisionTree 3 N
        python dd.py test.data SVM 4 Y

```

## Some images of the result

![knn](https://github.com/kbckbc/washu_fl22_cse514/blob/main/train_model/knn.png)
![rf](https://github.com/kbckbc/washu_fl22_cse514/blob/main/train_model/rf.png)
![dt](https://github.com/kbckbc/washu_fl22_cse514/blob/main/train_model/dt.png)
![svm](https://github.com/kbckbc/washu_fl22_cse514/blob/main/train_model/svm.png)


## Lesson learned
+ For binary classification problems, I would choose Decision Tree. Because the accuracy and speed were superior to other models. For some problems, Random Forest and SVM showed better accuracy, but the difference was negligible. However, in terms of speed, Decision Tree showed much faster speed than other models. Comprehensively, in binary classification, Decision Tree showed the best performance.
+ If it is a multi-class classification problem, I would choose SVM. The 'poly' or 'rbf' kernels have shown very high accuracy in multi-class classification problems. Unexpectedly, the kNN model also showed high accuracy, which is guessed because it was limited to 6 classes. If more classes are to be classified, I assume that SVM will show better results than kNN. SVM took a little longer to classify, but the test showed that SVM was the most accurate.
+ Dimensional reduction decreased the accuracy of most models, but the execution speed was significantly increased. In particular, in multi-class classification problems, the accuracy is significantly reduced. If the loss of accuracy outweighs the speed benefit of dimensionality reduction, then dimensionality reduction doesn't seem like a good choice. In the end, it seems that it is wise to decide whether to apply dimensionality reduction or not according to the given situation.
+ Due to this result, it was obtained that Decision Tree is better for binary classification and SVM is better for multi-class classification. Therefore, if I need to analyze new data, I will train the model to find the optimal hyper parameter with Decision Tree and SVM without testing other models.
