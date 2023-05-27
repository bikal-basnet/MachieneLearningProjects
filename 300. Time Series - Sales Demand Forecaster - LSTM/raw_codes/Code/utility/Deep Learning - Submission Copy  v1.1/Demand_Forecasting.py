#### Load Librarires
import h5py
import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import sys


#### Create  DataSet for training : x = no. of product sold at time t, Y = no. of  product sold at time t+1
## e.g Converts  [112,118,132,129,.........] TO
# X		Y
# 112		118
# 118		132
# 132		129

# Class DataImporter
# returns pandas.DataFrame
def load_csv_data(filepath, use_column = 0, skipfooter = 0 ):
    return pandas.read_csv(filepath, usecols = [use_column], engine = 'python', skipfooter = skipfooter).dropna( how= 'any')

# Class TsdataConvertor
def create_ts_dataset(dataset, look_back = 1 ):
    dataX, dataY = [] , []
    for i in range (len(dataset) - look_back - 1 ):
        a = dataset[i: (i+ look_back), 0 ]
        dataX.append(a)
        dataY.append(dataset[ i+look_back , 0])
    return numpy.array(dataX), numpy.array(dataY)


# Class TsDataHandler
# returns ndarray
def convert_to_float(dataframe):
    dataset = dataframe.values
    # Convert the data frame to ndarray as expected by the LSTM
    dataset = dataset.astype('float32')
    #print(dataframe)
    #print(dataset)
    #print(type(dataframe), type(dataset))
    return(dataset)

# reshape into X=t and Y=t+1
def convert_data_to_TS(data, look_back = 1):
    dataX, dataY = create_ts_dataset(data, look_back)
    # LSTM expects the input data in form : [samples, time steps, features]. Convert the train and test data
    dataX = numpy.reshape(dataX, (dataX.shape[0], 1, dataX.shape[1]))
    return dataX, dataY

# Becasue :LSTM are sensitive to scale, especially when sigmoid or tanh activation functyion is use.
def data_scaler():
    return MinMaxScaler(feature_range = (0,1))

def  normalise_ts_data(ndarray_dataset):
    scaler = data_scaler
    return scaler.fit_transform(ndarray_dataset)

# END :: Class TsDataHandler

# Class DataSplitter
# default 2/3 as train data
# return train, test data
def test_train_data_splitter(dataset, train_data_size_in_percentage = 0.67):
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    return dataset[0:train_size, :], dataset[train_size:len(dataset), :]

# Class TsModeler
from keras.layers import Dropout, activations
def get_ts_model( trainX, trainY, look_back = 1, nb_epochs = 100 ):
    model = Sequential()
    # takes input array of shape (*, 1) where (2,1) - (row,col) array example looks like [23]
    # 																					 [43]
    model.add(LSTM(20, input_shape=(None , look_back) ))
    #model.add(LSTM(20,  batch_input_shape=(None, None, look_back), return_sequences= True ))
    #print(model.summary)
    model.add( Dense(1) )
    model.add(Dense(1))
    model.add(Dense(1))
    model.add(Dense(1))
    model.add(Dense(1))
    model.add(Dense(1))
    #model.add(LSTM(1, return_sequences= True))
    #model.add(LSTM(1))
    # outputs array of shape (*,1)
    #model.add(Dense(1))
    #model.compile(loss='mean_absolute_error', optimizer='SGD')  # mape
    #model.compile(loss='poisson', optimizer='adam')  # mape
    model.compile( loss =  'mean_squared_error', optimizer = 'adam' ) # values closer to zero are better.
    #model.compile(loss='mean_squared_error', optimizer='adagrad')
    # Values of MSE are used for comparative purposes of two or more statistical meythods. Heavily weight outliers,  i.e weighs large errors more heavily than the small ones.
    # "In cases where this is undesired, mean absolute error is used.
    # REF: Available loss functions  https://keras.io/objectives.
    print('Start : Training model')
    # default  configuration
    model.fit(trainX, trainY, nb_epoch=nb_epochs, batch_size=1, verbose=2)
    #model.fit(trainX, trainY, nb_epoch=100, batch_size=1, verbose=2)
    print('Ends : Training Model')
    return model


def rmse(actual, pred):
    squared_diff = 0
    for i in range(0, len(pred) ):
        squared_diff =  squared_diff  + math.pow( ( pred[i] - actual[i] ), 2)
    rmse = math.sqrt( (squared_diff / ( len(pred)  ) ) )
    print('root mean squared error is', rmse)
    return rmse
    #sys.exit(1)

import time
import numpy as np
from array import array
# Class Evaluator
def evaluate_performance_model_based(trainX, trainY, model, scaler):
    ####  Performance Evaluation
    print('Evaluate starts')
    trainScore = model.evaluate(trainX, trainY, verbose=2)
    trainScore = math.sqrt(trainScore)
    #trainScore = scaler.inverse_transform(numpy.array([[trainScore]]))
    # Test Performanace
    testScore = model.evaluate(testX, testY, verbose=0)
    testScore = math.sqrt(testScore)
    #testScore = scaler.inverse_transform(numpy.array([[testScore]]))
    print('RMSE Train Score : %.4f . RMSE Test Score : %.4f' % (trainScore, testScore))


def convert_actual_to_1darray(actual):
    act = []
    for i in range(0, len(actual)):
        act.append(actual[i][0][0])
    return act

def convert_pred_to_1darray(prediction):
    pred = []
    for i in range(0, len( prediction ) ):
        pred.append( prediction[i][0] )
    return pred

def evaluate_performance(actual, prediction, model, scaler):
    ####  Performance Evaluation
    print('Evaluate starts')
    print('Training ', actual[0][0][0],' pred is ', prediction[0][0])
    print('Training 1 ', actual[1][0][0], ' pred is ', prediction[1][0])
    #sys.exit(1)
    squared_diff = 0
    act = convert_actual_to_1darray(actual)
    pred = convert_pred_to_1darray(prediction)
    rmseScore = rmse( act, pred )
    return rmseScore


import statsmodels.api as sm
def describe_ts_data(ts_dataset):
    #print(ts_dataset)
    ts = ts_dataset #data['Value']
    print('time series data is ', ts)
    decompose = sm.tsa.seasonal_decompose(ts, freq = 12 ,model = 'multiplicative' )
    trend = decompose.trend
    seasonal = decompose.seasonal
    residual = decompose.resid

    plt.subplot(411)
    plt.plot(ts_dataset, label = 'Product-A Sales')
    plt.legend(loc='best')

    plt.subplot(412)
    plt.plot(trend, label = 'Trend ')
    plt.legend(loc='best')

    plt.subplot(413)
    plt.plot(seasonal, label='Seasonality')
    plt.legend(loc='best')

    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')

    plt.tight_layout()

    print('trend is ', trend)
    print('seasonality is', seasonal)
    print('residual is', residual)
    plt.show()
    #sys.exit(1)
    print('Ts data statictics here')
    print('Minimum Product Sales Qty',min(ts_dataset))
    print('Maximum Product Sales Qty', max(ts_dataset))
    print('Average Product Sales Qty', numpy.average(ts_dataset))
    print('Standard Deviation of Product Sales Qty', numpy.std(ts_dataset))
    #sys.exit(1)

import numpy as np
import matplotlib.dates as mdates
def plot_ts_data(inputfilepath):
    days,  time = np.loadtxt(inputfilepath, delimiter = ',', unpack = True, skiprows = 3,
                             converters = {0: mdates.strpdate2num('"%m-%d"')})
    plt.plot_date(x = days, y = time, fmt = 'r-')
    plt.title("Monthly Sales of Plastic Manufacturer's Product-A")
    plt.xlabel('Date-Time')
    plt.ylabel('Sales Qty')
    plt.grid(True)
    plt.show()
    #sys.exit(1)


# Class PreProcessor
def movingaverage(interval, window_size):
    window= numpy.ones(int(window_size))/float(window_size)
    return numpy.convolve(interval, window, 'same')

def convert_nx1array_to_nlist(nx1array):
    nlist = []
    for val in nx1array:
        #print val[0]
        nlist.append(val[0])
    return nlist

def convert_nlist_to_nx1array(test_data_moving_average):
    train_nx1array = []
    for val in test_data_moving_average:
        train_nx1array.append([val])
        #print 'v', val
    return numpy.asarray(train_nx1array, dtype= float)

def remove_outlier(dataframe, reject_all_after_std_deviation):
    print('mean ',dataframe.median(), 'std', dataframe.std() )
    return dataframe[np.abs(dataframe - dataframe.median()) <= (reject_all_after_std_deviation * dataframe.std()) ].dropna()

def remove_negative_numbers(dataframe):
    return dataframe[ dataframe  >= 0 ].dropna()

# Class PredictonGenerator
def generate_predictions(use_ts_model, for_data):
    return use_ts_model.predict(for_data)

def plot_predictions(original_data, prediction_on_train_data, prediction_on_test_data, look_back, train_data_transformed = None ):
    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(original_data)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(prediction_on_train_data) + look_back, :] = prediction_on_train_data
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(original_data)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(prediction_on_train_data) + (look_back * 2) + 1:len(original_data) - 1, :] = prediction_on_test_data

    # plot baseline and predictions
    plt.title(' Actual Sales Demand vs Model Prediction Plot')
    plt.plot( original_data, label = "Actual Data" )
    plt.plot( trainPredictPlot, label = "Prediction on train data")
    plt.plot(testPredictPlot,"r-" , label = "Prediction on test data")
    if train_data_transformed != None:
        plt.plot(train_new, "r*--", label="Modified Train Data-MA")
    plt.xlabel('Date-Time Progression ')
    plt.ylabel('Sales Demand (Normalised to  0-1) ')
    plt.legend(loc = "lower right", shadow = True)
    plt.grid(True)
    plt.show()


# Implementation
if __name__ == "__main__":
    #### Reproducibility : Fix random seed
    numpy.random.seed(7)
    inputfilepath = 'Data/monthly-sales-of-product-a-for-a-Corrected.csv'
    dataframe = load_csv_data(filepath = inputfilepath, use_column=1, skipfooter=0)
    dataframe = remove_negative_numbers(remove_outlier(dataframe,3))
    dataset = convert_to_float(dataframe)
    data_scaler = data_scaler()
    ## Uncomment the describe ts dataset to see the trend, seasonality and the irregularity component of the dataset
    #describe_ts_data(dataset)
    dataset = normalise_ts_data(dataset)
    train, test = test_train_data_splitter(dataset, 0.67)
    original_train = train
    train_new = convert_nx1array_to_nlist(train)
    train_data_moving_average = movingaverage(train_new, 3)
    train_new =  convert_nlist_to_nx1array(train_data_moving_average)
    #### Convert  training and test data to Time series expected Data format
    look_back = 1
    trainX, trainY = convert_data_to_TS(train_new)
    testX, testY = convert_data_to_TS(test)
    model = get_ts_model(trainX, trainY, look_back, nb_epochs=100)
    trainPredict = generate_predictions(model, trainX)

    testPredict = generate_predictions(model, testX)
    trainScore = evaluate_performance(trainX, trainPredict, None, None)
    testScore = evaluate_performance(testX, testPredict, None, None)
    print('Train RMSE  Score : %.4f. Test RMSE Score : %.4f. ' % ( trainScore, testScore ))

    # Visualisation
    plot_predictions(dataset,trainPredict, testPredict, look_back, train_new  )


