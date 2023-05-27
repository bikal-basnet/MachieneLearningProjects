# LSTM for international airline passengers problem with regression framing
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


#### Create  DataSet for training : x = no. of passengers at time t, Y = no. of passengers at time t+1
## e.g Converts  [112,118,132,129,.........] TO
# X		Y
# 112		118
# 118		132
# 132		129

# Class DataImporter
# returns pandas.DataFrame
def load_csv_data(filepath, use_column = 0, skipfooter = 0 ):
	return pandas.read_csv(filepath, usecols = [use_column], engine = 'python', skipfooter = skipfooter)

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
def get_ts_model( trainX, trainY, look_back = 1 ):
	model = Sequential()
	# takes input array of shape (*, 1) where (2,1) - (row,col) array example looks like [23]
	# 																					 [43]
	#model.add(LSTM(5, input_shape=(36 , look_back) ))
	model.add(LSTM(1, batch_input_shape=(None, None, look_back), return_sequences= True ))
	#print(model.summary)
	#model.add(LSTM(1, return_sequences= True))
	model.add(LSTM(1))

	# outputs array of shape (*,1)
	model.add(Dense(1))

	#model.compile(loss='mean_absolute_error', optimizer='SGD')  # mape
	model.compile( loss =  'mean_squared_error', optimizer = 'adam' ) # values closer to zero are better.
	#model.compile(loss='mean_squared_error', optimizer='adagrad')
	# Values of MSE are used for comparative purposes of two or more statistical meythods. Heavily weight outliers,  i.e weighs large errors more heavily than the small ones.
	# "In cases where this is undesired, mean absolute error is used.
	# REF: Available loss functions  https://keras.io/objectives.
	print('Start : Training model')
	#model.fit(trainX, trainY, nb_epoch=50, batch_size=1, verbose=2)
	model.fit(trainX, trainY, nb_epoch=100, batch_size=1, verbose=2)
	print('Ends : Training Model')
	return model

import time
# Class Evaluator
def evaluate_performance(trainX, trainY, model, scaler):
	####  Performance Evaluation
	print('Evaluate starts')
	time.sleep(1)
	trainScore = model.evaluate(trainX, trainY, verbose=2)
	trainScore = math.sqrt(trainScore)
	#trainScore = scaler.inverse_transform(numpy.array([[trainScore]]))
	# Test Performanace
	testScore = model.evaluate(testX, testY, verbose=0)
	testScore = math.sqrt(testScore)
	#testScore = scaler.inverse_transform(numpy.array([[testScore]]))
	print('RMSE Train Score : %.4f . RMSE Test Score : %.4f' % (trainScore, testScore))

# Class PredictonGenerator
def generate_predictions(use_ts_model, for_data):
	return use_ts_model.predict(for_data)

def plot_predictions(original_data, prediction_on_train_data, prediction_on_test_data, look_back):
	# shift train predictions for plotting
	trainPredictPlot = numpy.empty_like(original_data)
	trainPredictPlot[:, :] = numpy.nan
	trainPredictPlot[look_back:len(prediction_on_train_data) + look_back, :] = prediction_on_train_data
	# shift test predictions for plotting
	testPredictPlot = numpy.empty_like(original_data)
	testPredictPlot[:, :] = numpy.nan
	testPredictPlot[len(prediction_on_train_data) + (look_back * 2) + 1:len(original_data) - 1, :] = prediction_on_test_data

	# plot baseline and predictions
	plt.title('DataSet: Blue Line. Prediction on train data: Green. Prediction on Test: Red')
	plt.plot(original_data)
	plt.plot(trainPredictPlot)
	plt.plot(testPredictPlot)
	plt.show()

	#plot baseline and predictions
	# plt.figure(1)
	# plt.subplot(311)
	# plt.title('Dataset Plot')
	# plt.plot(dataset)
	# plt.subplot(312)
	# plt.title('Training Prediction Plot')
	# plt.plot(trainPredictPlot)
	# plt.subplot(313)
	# plt.title('Test Prediction Plot')
	# plt.plot(testPredictPlot)
	# plt.show()

def describe_ts_data(ts_dataset):
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



# Implementation
if __name__ == "__main__":
	#### Reproducibility : Fix random seed
	numpy.random.seed(7)
	# dataframe = load_csv_data(filepath='Data/international-airline-passengers1.csv', use_column=1,	  skipfooter=0)
	# dataframe = load_csv_data(filepath = 'Data/Wallmart Data.csv', use_column = 1 , skipfooter = 0)
	inputfilepath = 'Data/monthly-sales-of-product-a-for-a-Corrected.csv'
	dataframe = load_csv_data(filepath = inputfilepath, use_column=1, skipfooter=0)
	#plot_ts_data(inputfilepath)
	dataset = convert_to_float(dataframe)
	data_scaler = data_scaler()
	describe_ts_data(dataset)

	dataset = normalise_ts_data(dataset)
	train, test = test_train_data_splitter(dataset, 0.67)

	#### Convert  training and test data to Time series expected Data format
	look_back = 1
	trainX, trainY = convert_data_to_TS(train)
	testX, testY = convert_data_to_TS(test)

	print('Training Data is', trainX, ' Training Data shape is' ,  trainX.shape)
	model = get_ts_model(trainX, trainY, look_back)
	#modelfilename = 'PredictionModels/keras_model_retail.h5'
	#model.save(modelfilename)
	#del model
	# return a compiled model, identical to the previous one
	#model = load_model(modelfilename)
	evaluate_performance(trainX, trainY, model, data_scaler)
	trainPredict = generate_predictions(model, trainX)

	# for i in range(0, len( trainPredict )):
	# 	print ('Training Data. Actual :: ',trainX[i][0][0]*100, ' .Pred :: ',trainPredict[i][0]*100,' .Error ::', (trainPredict[i][0]- trainX[i][0][0]) *100 )
	testPredict = generate_predictions(model, testX)
	plot_predictions(dataset, trainPredict, testPredict, look_back)


