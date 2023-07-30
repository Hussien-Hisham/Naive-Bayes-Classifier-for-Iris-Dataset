from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from random import randrange
import pandas as pd

### Load Data ##
iris = load_iris()
### load independent variable data ###
X = iris.data
### load dependent variable data ###
Y = iris.target

#### Split Test Data and Train Data ###
def train_test_split(X, Y, Trainratio=0.7, random_state=8):
    x_train = list()
    y_train= list()
    DATA_size = Trainratio * len(X)
    dataset_copy = list(X)
    labelset_copy= list(Y)
    while len(x_train) < DATA_size:
        index = randrange(0, len(dataset_copy),random_state)
        x_train.append(dataset_copy.pop(index))
        y_train.append(labelset_copy.pop(index))
    x_test=dataset_copy.copy()
    y_test=labelset_copy.copy()
    return x_train,x_test,y_train ,y_test

### Calculate Acurracy Func ####
def calculate_accuracy(y_pred,y_test):
 return (100-((((y_test != y_pred).sum())/x_test.shape[0])*100))

x_train, x_test, y_train , y_test= train_test_split(X, Y)
gnb = GaussianNB()
y_pred = gnb.fit(x_train, y_train).predict(x_test)
x_test = pd.DataFrame(x_test)
### main ###
def main():
    if __name__== "__main__" :
        print(calculate_accuracy(y_pred, y_test))
main()