from pylab import plot, ylim, xlim, show, xlabel, ylabel
from numpy import linspace, loadtxt
import numpy

data = loadtxt("data/sunspots.txt", float, skiprows=1, delimiter=',')

def movingaverage(interval, window_size):
    window= numpy.ones(int(window_size))/float(window_size)
    return numpy.convolve(interval, window, 'same')

x = data[:,0]
y = data[:,1]

print y
#sys.exit(1)
test_data = numpy.array([[ 0.0478723 ],
       [ 0.04680848],
       [ 0.21170211],
       [ 0.27021271]
       ])
print(test_data)

def convert_nx1list_to_nlist(nx1list):
    nlist = []
    for val in nx1list:
        print val[0]
        nlist.append(val[0])
    return nlist

test_data = convert_nx1list_to_nlist(test_data)

test_data_moving_average = movingaverage(test_data, 2)
print(test_data_moving_average, 'test_data_moving_average size is ', test_data_moving_average.size)

nx1array = []
for val in test_data_moving_average:
    nx1array.append([val])
    print 'v', val
print(nx1array)

numpy.asarray(nx1array, dtype= float)

print('convert to ndarray', numpy.asarray(test_data_moving_average, dtype = float ) )

import sys
sys.exit(1)
y_av = movingaverage(y, 4)

plot(x,y,"b-")
y_av = movingaverage(y, 4)
plot(x, y_av,"r")
xlim(1700,2015)
xlabel("Months since Jan 1749.")
ylabel("No. of Sun spots")
#grid(True)
show()