from __future__ import print_function
from sklearn import preprocessing

#!/usr/bin/env python
# Example script for recurrent network usage in PyBrain.
__author__ = "Martin Felder"
__version__ = '$Id$'

data = [1]* 3 + [2] * 3
data *= 3
#print(data)

def normalise(data):
    return [round(float(i * 1) / max(data), 2) for i in data]

#data1 = [112,118,132,129,121,135,148,148,136,119,104,118]
######
## Test with REtail Data
######
month   = [ 1 , 2 , 3 , 4 ,  5 , 6 ,  7 ,  8 , 9 , 10 , 11 , 12 ]
data1 = normalise([112,118,132,129,121,135,148,148,136,119,104,118])
data2 = normalise([115,126,141,135,125,149,170,170,158,133,114,140])
data3 = normalise([145,150,178,163,172,178,199,199,184,162,146,166])
data4 = normalise([171,180,193,181,183,218,230,242,209,191,172,194])

# data1 = [ 1 , 1 , 2 , 4 , 37 , 1 , 66 , 32 , 7 , 12 , 34 ,  4 ]
# data2 = [ 2 , 2 , 3 , 4 , 40 , 4 , 70 , 37 , 2 , 10 , 24 ,  1 ]
# data1 = [round(float(i*1)/max(data1),2) for i in data1]

#print('data1 is ', data1)
# data2 = [round(float(i*1)/max(data2),2) for i in data2]
#data2 = [round(float(i*1.12),2) for i in data1]

# data2 = [1,1,12,44,22,50,235,74,51,49,37,70]
# data2 = [float(i*100)/max(data2) for i in data2]
#
# data3 = [50,96,97,67,124,326,101,120,82,103,194,23]
# data3 = [float(i*100)/max(data3) for i in data3]

data = []
data.extend(data1)
data.extend(data2)
data.extend(data3)
data.extend(data4)
# data.extend([data1[i]*2 for i in range(len(data1))])
# data.extend([data1[i]*3 for i in range(len(data1))])
# data.extend([data1[i]*4 for i in range(len(data1))])

#data = [float(i*1)/max(data) for i in data]
#data = [24924.5,46039.49,41595.55,19403.54,21827.9,21043.39,22136.64,26229.21,57258.43,42960.91,17596.96,16145.35,16555.11,17413.94,18926.74,14773.04,15580.43,17558.09,16637.62,16216.27,16328.72,16333.14,17688.76,17150.84,15360.45,15381.82,17508.41,15536.4,15740.13,15793.87,16241.78,18194.74,19354.23,18122.52,20094.19,23388.03,26978.34,25543.04,38640.93,34238.88,19549.39,19552.84,18820.29,22517.56,31497.65,44912.86,55931.23,19124.58,15984.24,17359.7,17341.47,18461.18,21665.76,37887.17,46845.87,19363.83,20327.61,21280.4,20334.23,20881.1,20398.09,23873.79,28762.37,50510.31,41512.39,20138.19,17235.15,15136.78,15741.6,16434.15,15883.52,14978.09,15682.81,15363.5,16148.87,15654.85,15766.6,15922.41,15295.55,14539.79,14689.24,14537.37,15277.27,17746.68,18535.48,17859.3,18337.68,20797.58,23077.55,23351.8,31579.9,39886.06,18689.54,19050.66,20911.25,25293.49,33305.92,45773.03,46788.75,23350.88,16567.69,16894.4,18365.1,18378.16,23510.49,36988.49,54060.1,20124.22,20113.03,21140.07,22366.88,22107.7,28952.86,57592.12,34684.21,16976.19,16347.6,17147.44,18164.2,18517.79,16963.55,16065.49,17666,17558.82,16633.41,15722.82,17823.37,16566.18,16348.06,15731.18,16628.31,16119.92,17330.7,16286.4,16680.24,18322.37,19616.22,19251.5,18947.81,21904.47,22764.01,24185.27,27390.81]
print(data)
#exit(0)

#data =[-0.025752496,0.091349779,0.112477983,-0.043485112,-0.076961041,0.175632569,0.131852131,0.000000000,-0.073203404,-0.172245905,-0.154150680,0.205443974]
#data = data * 100

from pybrain.datasets import SequentialDataSet
from itertools import cycle

ds = SequentialDataSet(1, 1)
for sample,next_sample in zip( data, cycle(data[1:] ) ):
    ds.addSample(sample, next_sample)
print(ds)

from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import LSTMLayer



# Buils a simple LSTM network with 1 input node, 1 output node and 5 LSTM cells
net = buildNetwork(1, 12, 1, hiddenclass=LSTMLayer, peepholes = False, outputbias=False, recurrent=True)
# net = buildNetwork(1, 1, 1, hiddenclass=LSTMLayer, peepholes = True, outputbias=False, recurrent=True)
# rnn = buildNetwork( trndata.indim, 5, trndata.outdim, hiddenclass=LSTMLayer, outclass=SoftmaxLayer, outputbias=False, recurrent=True)

from pybrain.supervised import RPropMinusTrainer
from sys import stdout

trainer = RPropMinusTrainer(net, dataset=ds, verbose = True)
#trainer.trainUntilConvergence()

train_errors = [] # save errors for plotting later
EPOCHS_PER_CYCLE = 100            # increasing the epochs to 20, decreases accuracy drastically,  decreasing epochs is desiredepoch # 5 err = 0.04
CYCLES = 10                   # vary the epochs adn the cycles and the LSTM cells to  get more accurate results.
EPOCHS = EPOCHS_PER_CYCLE * CYCLES
for i in xrange(CYCLES):
    trainer.trainEpochs(EPOCHS_PER_CYCLE)     # train on the given data set for given number of epochs
    train_errors.append(trainer.testOnData())
    epoch = (i+1) * EPOCHS_PER_CYCLE
    print("\r epoch {}/{}".format(epoch, EPOCHS), end="")
    stdout.flush()

print()
print("final error =", train_errors[-1])


## Plot  the data and the training
import matplotlib.pyplot as plt
plt.plot(range(0, EPOCHS, EPOCHS_PER_CYCLE), train_errors)
plt.xlabel('epoch')
plt.ylabel('error')
plt.show()
mape_error = 0
count = 0
## Predict new examples
for sample, actual in ds.getSequenceIterator(0):
    PredictedValue = net.activate(sample)
    count = count + 1

    # MAPE Error
    CurrentError = abs ((actual - PredictedValue) * 100.00 / actual )
    mape_error = mape_error +  CurrentError
    print("   sample = %4.3f. Prediction = %4.3f.  Actual  = %4.3f. Error = %4.3f. Normalised Error  = %4.3f " % (sample, PredictedValue,  actual, (actual - PredictedValue  ), CurrentError ) )

print ("Total Mean Absolute Percentage Error = %4.3f Percentage" % (mape_error/count) )

