# washu_fl22_cse514
22 Fall semester CSE 523(Data mining) course works at Washington University in St. Louis.


# Programming Assignment #1
[pdf](https://github.com/kbckbc/washu_fl22_cse514/blob/main/linear_regression/Programming%20Assignment%201.pdf)


+ This project is to find a linear function on some data sets using Linear regression methodology and gradient descent algorithm for optimization. Linear regression is a statistical technique that models the linear relationship between a dependent variable y and one or more independent variables x. The gradient descent refers to changes in the model moving along the slope or slope of the graph toward the lowest possible error value. 
+ The data set is about the strength of concrete when concrete is mixed up with several ingredients. In the data set, 8 feature values affect the strength of concrete. 
+ More specifically, I’m going to build a program to figure out the proper linear model by adopting Uni-variate and Multi-variate regression. Uni-variate linear regression is y = mx + b and multi-variate is y = b0 + m1*x1 + m2*x2 + .. + mn*xn. And I will use MSE(Mean Squared Error) as a loss function. The program can use one feature from the data set or all 8 features to find a linear function. It can be done by designating parameters.

## how to run
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


# Programming Assignment #2
[pdf](https://github.com/kbckbc/washu_fl22_cse514/blob/main/train_model/Programming%20Assignment%202.pdf)



## how to run
```
```