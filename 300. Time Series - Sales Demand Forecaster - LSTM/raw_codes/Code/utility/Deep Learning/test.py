#############
## Data selection
###############


import pandas
dataframe = pandas.read_csv('Data/international-airline-passengers.csv', engine = 'python', skipfooter = 3)

Id = dataframe.iloc[0,0]

print dataframe.iloc[:,2]

# print dataframe.Value
# dataframe1 = dataframe.Value
# dataset = dataframe1.values
# dataset = dataset.astype('float32')
#
modelfilename = 'PredictionModels/keras_model'+str(Id)+'.h5'
print('model file name is ',modelfilename)


#dataset = dataframe.values
#dataset = dataset.astype('float32')
#print dataset