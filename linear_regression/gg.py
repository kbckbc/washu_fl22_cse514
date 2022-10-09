from contextlib import nullcontext
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
import time
import sys


# MyLinearUni is a uni-variate Linear Regression simulate class
class MyLinearUni:
    def __init__(self, loss_type, x, y, coeff, alpha = 0.001, repeat = 4000):
        """ 
            x: feature
            y: result
            alpha: learning rate, default is 0.001
            repeat: use this as a stop condition
            b0, b1: coefficient
        """
        self.loss_type = loss_type
        self.i = 0
        self.x = x
        self.y = y
        self.alpha = alpha
        self.repeat = repeat
        self.b0 = coeff[0]
        self.b1 = coeff[1]

        if len(self.x) != len(self.y):
            raise TypeError("x and y should have same number of rows.")
  
    def predict(self, x):
        b0, b1 = self.b0, self.b1
        # Y = b0 + b1 * X
        return b0 + b1 * x
  
    def loss_derivative_mse(self, i):
        x, y = self.x, self.y
        predict = self.predict

        sum = 0
        for xi,yi in zip(x,y):
            if i==0:
                sum += 2*(predict(xi) - yi) * 1
            else:
                sum += 2*(predict(xi) - yi) * xi
        return sum/len(x)

    def loss_derivative_mae(self, i):
        x, y = self.x, self.y
        predict = self.predict

        sum = 0
        for xi,yi in zip(x,y):
            if i==0:
                sum += 1 if predict(xi) > yi else -1
            else:
                sum += xi if predict(xi) > yi else -xi
        # print('mae', sum, len(x), sum/len(x))
        return sum/len(x)
  
    def update_coeff(self, i):
        alpha = self.alpha
        if self.loss_type == 'mse':
            loss_derivative = self.loss_derivative_mse
        else:
            loss_derivative = self.loss_derivative_mae

        if i == 0:
            self.b0 -= alpha * loss_derivative(i)
        elif i == 1:
            self.b1 -= alpha * loss_derivative(i)
  
    def stop(self, repeat):
        if self.i == repeat:
            self.i += 1
            return True
        else:
            self.i += 1
            return False
  
    def training(self):
        update_coeff = self.update_coeff
        self.i = 0
        while True:
            if self.stop(self.repeat):
                break
            else:
                update_coeff(0)
                update_coeff(1)

                # print('train', self.b0, self.b1)
  
# MyLinearMulti is a multi-variate Linear Regression simulate class  
class MyLinearMulti:
    def __init__(self, loss_type, train, y, alpha = 0.001, repeat = 4000):
        """ 
            x: feature
            y: result
            alpha: learning rate, default is 0.001
            repeat: use this as a stop condition
            b0, b1: coefficient
        """
        self.loss_type = loss_type
        self.i = 0
        self.y = y
        self.x1 = train[0]
        self.x2 = train[1]
        self.x3 = train[2]
        self.x4 = train[3]
        self.x5 = train[4]
        self.x6 = train[5]
        self.x7 = train[6]
        self.x8 = train[7]
        self.alpha = alpha
        self.repeat = repeat
        self.b0 = 0
        self.b1 = 0
        self.b2 = 0
        self.b3 = 0
        self.b4 = 0
        self.b5 = 0
        self.b6 = 0
        self.b7 = 0
        self.b8 = 0
        if len(self.x1) != len(self.y):
            raise TypeError("x and y should have same number of rows.")
  
    def predict(self, x1, x2, x3, x4, x5, x6, x7, x8):
        b0,b1,b2,b3,b4,b5,b6,b7,b8 = self.b0,self.b1,self.b2,self.b3,self.b4,self.b5,self.b6,self.b7,self.b8

        # Y = b0 + b1*x1 + b2*x2 + b3*x3 + b4*x4 + b5*x5 + b6*x6 + b7*x7 + b8*x8 
        return b0 + b1*x1 + b2*x2 + b3*x3 + b4*x4 + b5*x5 + b6*x6 + b7*x7 + b8*x8 
  
    def loss_derivative_mse(self, i):
        y = self.y
        x1,x2,x3,x4,x5,x6,x7,x8 = self.x1,self.x2,self.x3,self.x4,self.x5,self.x6,self.x7,self.x8
        predict = self.predict
      
        ss = 0
        for xx1,xx2,xx3,xx4,xx5,xx6,xx7,xx8,yi in zip(x1,x2,x3,x4,x5,x6,x7,x8,y):
            if i==0:
                ss += 2*(predict(xx1,xx2,xx3,xx4,xx5,xx6,xx7,xx8) - yi) * 1
            elif i==1:
                ss += 2*(predict(xx1,xx2,xx3,xx4,xx5,xx6,xx7,xx8) - yi) * xx1
            elif i==2:
                ss += 2*(predict(xx1,xx2,xx3,xx4,xx5,xx6,xx7,xx8) - yi) * xx2
            elif i==3:
                ss += 2*(predict(xx1,xx2,xx3,xx4,xx5,xx6,xx7,xx8) - yi) * xx3
            elif i==4:
                ss += 2*(predict(xx1,xx2,xx3,xx4,xx5,xx6,xx7,xx8) - yi) * xx4
            elif i==5:
                ss += 2*(predict(xx1,xx2,xx3,xx4,xx5,xx6,xx7,xx8) - yi) * xx5
            elif i==6:
                ss += 2*(predict(xx1,xx2,xx3,xx4,xx5,xx6,xx7,xx8) - yi) * xx6
            elif i==7:
                ss += 2*(predict(xx1,xx2,xx3,xx4,xx5,xx6,xx7,xx8) - yi) * xx7
            elif i==8:
                ss += 2*(predict(xx1,xx2,xx3,xx4,xx5,xx6,xx7,xx8) - yi) * xx8
        return ss/len(x1)
 
    def loss_derivative_mae(self, i):
        y = self.y
        x1,x2,x3,x4,x5,x6,x7,x8 = self.x1,self.x2,self.x3,self.x4,self.x5,self.x6,self.x7,self.x8
        predict = self.predict
      
        sum = 0
        for xx1,xx2,xx3,xx4,xx5,xx6,xx7,xx8,yi in zip(x1,x2,x3,x4,x5,x6,x7,x8,y):
            if i==0:
                sum += 1 if predict(xx1,xx2,xx3,xx4,xx5,xx6,xx7,xx8) > yi else -1
            elif i==1:
                sum += xx1 if predict(xx1,xx2,xx3,xx4,xx5,xx6,xx7,xx8) > yi else -xx1
            elif i==2:
                sum += xx2 if predict(xx1,xx2,xx3,xx4,xx5,xx6,xx7,xx8) > yi else -xx2
            elif i==3:
                sum += xx3 if predict(xx1,xx2,xx3,xx4,xx5,xx6,xx7,xx8) > yi else -xx3
            elif i==4:
                sum += xx4 if predict(xx1,xx2,xx3,xx4,xx5,xx6,xx7,xx8) > yi else -xx4
            elif i==5:
                sum += xx5 if predict(xx1,xx2,xx3,xx4,xx5,xx6,xx7,xx8) > yi else -xx5
            elif i==6:
                sum += xx6 if predict(xx1,xx2,xx3,xx4,xx5,xx6,xx7,xx8) > yi else -xx6
            elif i==7:
                sum += xx7 if predict(xx1,xx2,xx3,xx4,xx5,xx6,xx7,xx8) > yi else -xx7
            elif i==8:
                sum += xx8 if predict(xx1,xx2,xx3,xx4,xx5,xx6,xx7,xx8) > yi else -xx8

        return sum/len(x1)

    def update_coeff(self, i):
        alpha = self.alpha
        if self.loss_type == 'mse':
            loss_derivative = self.loss_derivative_mse
        else:
            loss_derivative = self.loss_derivative_mae

        if i == 0:
            self.b0 -= alpha * loss_derivative(i)
        elif i == 1:
            self.b1 -= alpha * loss_derivative(i)
        elif i == 2:
            self.b2 -= alpha * loss_derivative(i)
        elif i == 3:
            self.b3 -= alpha * loss_derivative(i)
        elif i == 4:
            self.b4 -= alpha * loss_derivative(i)
        elif i == 5:
            self.b5 -= alpha * loss_derivative(i)
        elif i == 6:
            self.b6 -= alpha * loss_derivative(i)
        elif i == 7:
            self.b7 -= alpha * loss_derivative(i)
        elif i == 8:
            self.b8 -= alpha * loss_derivative(i)
  
    def stop(self, repeat):
        self.i += 1
        if self.i == repeat:
            return True
        else:
            return False
  
    def training(self):
        update_coeff = self.update_coeff
        self.i = 0
        while True:
            if self.stop(self.repeat):
                break
            else:
                update_coeff(0)
                update_coeff(1)
                update_coeff(2)
                update_coeff(3)
                update_coeff(4)
                update_coeff(5)
                update_coeff(6)
                update_coeff(7)
                update_coeff(8)

def mse(expected, prediction):
    diff = []
    for x, y in zip(expected, prediction):
        gap = y-x
        diff.append(gap ** 2)
    return sum(diff)/len(expected)

def mae(expected, prediction):
    diff = []
    for x, y in zip(expected, prediction):
        gap = y-x
        diff.append(abs(gap))
    return sum(diff)/len(expected)    

def var(loss_type, expected):
    diff = []
    mean = np.mean(expected)
    for item in expected:
        gap = item - mean
        if loss_type == 'mse':
            diff.append(gap**2)
        else:
            diff.append(abs(gap))
    return sum(diff)/len(expected)

def rsquare(loss_type, expected, prediction):
    if loss_type == 'mse':
        return 1 - (mse(expected, prediction)/var(loss_type, expected))
    else:
        return 1 - (mae(expected, prediction)/var(loss_type, expected))

def usage(exec_name):
    print('Usage:')
    print('\tpython %s [arg1] [arg2] [arg3] [arg4] [arg5]' % (exec_name))
    print('\targ1: input excel file')
    print('\targ2: train or test') 
    print('\t\ttrain - apply learned model to a training data set')
    print('\t\ttest - apply learned model to a test data set')
    print('\targ3: mse or mae') 
    print('\t\tmse - using mse as a loss function')
    print('\t\tmae - using mae as a loss function')
    print('\targ4: nopre or prezero or premean') 
    print('\t\tnopre - no pre-processing on the data')
    print('\t\tprezero - pre-processing data by deleting zero values')
    print('\t\tpremean - pre-processing data by substituing zero with mean value')
    print('\targ5: plot or noplot') 
    print('\t\tplot - show a plot on a uni-varite model')
    print('\t\tnoplot - not showing a plot on a uni-varite model')
    print('Example:')
    print('\tpython %s Concrete_Data.xls train mse nopre noplot' % (exec_name))
    print('\tpython %s Concrete_Data.xls train mse nopre plot' % (exec_name))
    print('\tpython %s Concrete_Data.xls train mae nopre noplot' % (exec_name))
    print('\tpython %s Concrete_Data.xls train mae nopre plot' % (exec_name))
    print('\tpython %s Concrete_Data.xls test mse nopre noplot' % (exec_name))
    print('\tpython %s Concrete_Data.xls test mse nopre plot' % (exec_name))
    print('\tpython %s Concrete_Data.xls test mae nopre noplot' % (exec_name))
    print('\tpython %s Concrete_Data.xls test mae nopre plot' % (exec_name))
    exit()

# return data_
def checkarg(argv):
    if len(argv) != 6:
        usage(argv[0])
    else :
        data_type = argv[2]
        loss_type = argv[3]
        pre_type = argv[4]
        plot_type = argv[5]

        if not (argv[2] == 'train' or argv[2] == 'test'):
            print('Error:')
            print('\tcheck the argument [%s]' % (argv[2]))
            usage(argv[0])

        if not (argv[3] == 'mse' or argv[3] == 'mae'):
            print('Error:')
            print('\tcheck the argument [%s]' % (argv[3]))
            usage(argv[0])

        if not (argv[4] == 'nopre' or argv[4] == 'prezero' or argv[4] == 'premean'):
            print('Error:')
            print('\tcheck the argument [%s]' % (argv[4]))
            usage(argv[0])

        if not (argv[5] == 'noplot' or argv[5] == 'plot'):
            print('Error:')
            print('\tcheck the argument [%s]' % (argv[5]))
            usage(argv[0])
        
        return [data_type, loss_type, pre_type, plot_type]



if __name__ == '__main__':

    # check program parameter
    ret_arg = checkarg(sys.argv)
    data_type = ret_arg[0]
    loss_type = ret_arg[1]
    pre_type = ret_arg[2]
    plot_type = ret_arg[3]

    # read by default 1st sheet of an excel file
    filename = sys.argv[1]
    df = pd.read_excel(filename,  sheet_name='Sheet1', usecols='A:I')

    # default value of coefficient
    # first of array is for Cement and second is for Blast ans so and so forth
    # adjust these value for proper result
    default_coeff_mse = [[0,0],[35,0],[33,0],[85,-8],[33,1],[80,-10],[40,0],[32,0]]
    default_coeff_mae = [[20,0],[30,0],[33,0],[70,-3],[25,1.5],[90,-3],[40,0],[32,0]]

    if loss_type == 'mse':
        coeff_uni = default_coeff_mse
        alpha_num = 0.000001
    else:
        coeff_uni = default_coeff_mae
        alpha_num = 0.00001

    # divide the data set by 900 and 130
    # 900 is traning data set
    # 130 is test data set
    divide_num = 900
    repeat_num = 4000

    # read data from excel
    x = []
    x.append(df[df.columns[0]].tolist())
    x.append(df[df.columns[1]].tolist())
    x.append(df[df.columns[2]].tolist())
    x.append(df[df.columns[3]].tolist())
    x.append(df[df.columns[4]].tolist())
    x.append(df[df.columns[5]].tolist())
    x.append(df[df.columns[6]].tolist())
    x.append(df[df.columns[7]].tolist())
    y = df[df.columns[8]].tolist() 


    # shuffle the data and divide them into train and test data set
    oldPair = list(zip(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],y))
    newPair = []
    pair = oldPair

    # pre-processing data
    # prezero is deleting all zeros in the dataset
    # premean is to substitue zero values with mean value
    if pre_type == 'prezero':
        for x0,x1,x2,x3,x4,x5,x6,x7,y in oldPair:
            if x0!=0 and x1!=0 and x2!=0 and x3!=0 and x4!=0 and x5!=0 and x6!=0 and x7!=0 :
                newPair.append([x0,x1,x2,x3,x4,x5,x6,x7,y])

        pair = newPair
        divide_num = round(len(newPair) * 0.1)
    elif pre_type == 'premean':
        for col in x:
            mm = np.mean(col)
            for i in range(len(col)):
                if col[i] == 0:
                    col[i] = mm

    random.shuffle(pair)
    train = pair[:divide_num]
    test = pair[divide_num:]

    # prepare for the train and test data
    x_train = []
    x_train.append([item[0] for item in train])
    x_train.append([item[1] for item in train])
    x_train.append([item[2] for item in train])
    x_train.append([item[3] for item in train])
    x_train.append([item[4] for item in train])
    x_train.append([item[5] for item in train])
    x_train.append([item[6] for item in train])
    x_train.append([item[7] for item in train])
    y_train = [item[8] for item in train]

    x_test = []
    x_test.append([item[0] for item in test])
    x_test.append([item[1] for item in test])
    x_test.append([item[2] for item in test])
    x_test.append([item[3] for item in test])
    x_test.append([item[4] for item in test])
    x_test.append([item[5] for item in test])
    x_test.append([item[6] for item in test])
    x_test.append([item[7] for item in test])
    y_test = [item[8] for item in test] 

    print('Linear regression')
    print('\toptimizer: Gradient descent algorithm')
    print('\tloss function: %s' % (loss_type))
    print('\ttotal data row:', len(pair))
    print('\trandomly divided by:',divide_num, 'vs', len(pair) - divide_num)
    print('\trepeat(stop condition):', repeat_num)
    print('\talpha(learning rate): %f'% (alpha_num))

    for i in range(9):
        # if i != 0:
        #     continue

        # allocate MyLinear class and train it
        start = time.time()
        if i != 8:
            myLinear = MyLinearUni(
                loss_type,
                x_train[i],
                y_train,
                coeff_uni[i],
                alpha_num,
                repeat_num
            )
        else:
            myLinear = MyLinearMulti(
                loss_type,
                x_train,
                y_train,
                alpha_num,
                repeat_num
            )
        myLinear.training()
        end = time.time()

        # get the prediction from a trained model
        if data_type == 'train':
            expect_data = y_train
            pred_data = []
            if i != 8:
                for item in x_train[i]:
                    pred_data.append(myLinear.predict(item))
            else:
                for x1,x2,x3,x4,x5,x6,x7,x8 in zip(x_train[0], x_train[1], x_train[2], x_train[3], x_train[4], x_train[5], x_train[6], x_train[7]) :
                    pred_data.append(myLinear.predict(x1,x2,x3,x4,x5,x6,x7,x8))

        elif data_type == 'test':
            expect_data = y_test
            pred_data = []
            if i != 8:
                for item in x_test[i]:
                    pred_data.append(myLinear.predict(item))
            else:
                for x1,x2,x3,x4,x5,x6,x7,x8 in zip(x_test[0], x_test[1], x_test[2], x_test[3], x_test[4], x_test[5], x_test[6], x_test[7]) :
                    pred_data.append(myLinear.predict(x1,x2,x3,x4,x5,x6,x7,x8))

        print("%s, %s: column[%d] time[%1.2f]: %s, variance, r^2 [%3.3f] [%3.3f] [%3.3f]" % (
            'uni' if i < 8 else 'mul',
            data_type,
            i, 
            end - start,
            loss_type,
            mse(expect_data, pred_data) if loss_type == 'mse' else mae(expect_data, pred_data), 
            var(loss_type, expect_data), 
            rsquare(loss_type, expect_data, pred_data),
            ))        

        if plot_type == 'plot' and i != 8:
            if data_type == 'train' :
                plt.scatter(x_train[i], expect_data, color="black")
                plt.plot(x_train[i], pred_data, color="blue", linewidth = 2)
                plt.show()
            else :
                plt.scatter(x_test[i], expect_data, color="black")
                plt.plot(x_test[i], pred_data, color="blue", linewidth = 2)
                plt.show()
  