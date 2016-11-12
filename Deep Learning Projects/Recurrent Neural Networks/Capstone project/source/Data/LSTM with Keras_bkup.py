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

#### Create  DataSet for training : x = no. of passengers at time t, Y = no. of passengers at time t+1
## e.g Converts  [112,118,132,129,.........] TO
# X		Y
# 112		118
# 118		132
# 132		129

def create_dataset(dataset, look_back = 1 ):
	dataX, dataY = [] , []
	for i in range (len(dataset) - look_back - 1 ):
		a = dataset[i: (i+ look_back), 0 ]
		dataX.append(a)
		dataY.append(dataset[ i+look_back , 0])
	return numpy.array(dataX), numpy.array(dataY)



#### Reproducibility : Fix random seed
numpy.random.seed(7)

#### Load Data
#dataframe = pandas.read_csv('Data/Wallmart Data.csv', usecols = [1], engine = 'python', skipfooter = 3)
dataframe = pandas.read_csv('Data/international-airline-passengers1.csv', usecols = [1], engine = 'python', skipfooter = 3)
#dataframe = pandas.read_csv('Data/international-airline-passengers.csv' , engine = 'python', skipfooter = 3)
#dataframe1 = dataframe.Value
dataset = dataframe.values
# Convert the data frame to ndarray as expected by the LSTM
dataset = dataset.astype('float32')
print(dataframe)
print(dataset)
print(type(dataframe),  type(dataset) )

import sys
#sys.exit()

#### Normalise Data
# Becasue :LSTM are sensitive to scale, especially when sigmoid or tanh activation functyion is use.
scaler = MinMaxScaler(feature_range = (0,1))
dataset = scaler.fit_transform(dataset)


#### Train-Test Data Split : 67% train , 33% test data size
train_size = int(len(dataset) * 0.67 )
test_size = len(dataset)- train_size
train, test = dataset[0:train_size , : ], dataset[ train_size:len(dataset), : ]
print('length of train and test is ', len(train), len(test))


#### Convert to Time series expected Data format
# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back )
testX, testY = create_dataset(test, look_back )

# LSTM expects the input data in form : [samples, time steps, features]. Convert the train and test data
trainX = numpy.reshape(trainX, (trainX.shape[0] , 1, trainX.shape[1] ))
testX  = numpy.reshape(testX, (testX.shape[0] , 1, testX.shape[1] ))

#### Train LSTM network
model = Sequential()
model.add( LSTM( 4, input_dim = look_back ) )
model.add( Dense(1) )
model.compile( loss =  'mean_absolute_error', optimizer = 'adam' ) # mape
#model.compile( loss =  'mean_squared_error', optimizer = 'adam' ) # values closer to zero are better.
# Values of MSE are used for comparative purposes of two or more statistical meythods. Heavily weight outliers,  i.e weighs large errors more heavily than the small ones.
# "In cases where this is undesired, mean absolute error is used.
#REF: Available loss functions  https://keras.io/objectives.
print('Start : Training model')
model.fit( trainX, trainY, nb_epoch = 100 , batch_size = 1, verbose = 2 )
print('Ends : Training Model')


model.save('PredictionModels/keras_model.h5')
del model

import time
#time.sleep(3)

# return a compiled model, identical to the previous one
model = load_model('PredictionModels/keras_model.h5')

####  Performance Evaluation 
trainScore = model.evaluate( trainX, trainY, verbose = 0 )
trainScore = math.sqrt( trainScore )
trainScore = scaler.inverse_transform( numpy.array( [[trainScore]] ) )
# Test Performanace
testScore = model.evaluate( testX, testY, verbose = 0 )
testScore = math.sqrt( testScore )
testScore = scaler.inverse_transform( numpy.array( [[testScore]] ) )
print('RMSE Train Score : %.2f . RMSE Test Score : %.2f' % ( trainScore, testScore ) )



# generate predictions for training
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(dataset)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
