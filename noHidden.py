from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer

from pybrain.structure import FeedForwardNetwork
n = FeedForwardNetwork()
from pybrain.structure import LinearLayer, SigmoidLayer, SoftmaxLayer
inLayer1 = LinearLayer(64)
hiddenLayer1 = SigmoidLayer(64)
outLayer = LinearLayer(10)
#inLayer2 = LinearLayer(2)
#hiddenLayer2 = SigmoidLayer(3)
n.addInputModule(inLayer1)
#n.addInputModule(inLayer2)
n.addModule(hiddenLayer1)
#n.addModule(hiddenLayer2)
n.addOutputModule(outLayer)
from pybrain.structure import FullConnection
n.addConnection(FullConnection(inLayer1, hiddenLayer1))
#n.addConnection(FullConnection(inLayer2, hiddenLayer2))
n.addConnection(FullConnection(hiddenLayer1, outLayer))
#n.addConnection(FullConnection(hiddenLayer2, outLayer))
#n.addConnection(FullConnection(inLayer1, outLayer))
n.sortModules()

#from pybrain.datasets import SupervisedDataSet
#alldata = SupervisedDataSet(64, 1)
#alldata.addSample((0, 0, 0, 0), (0,))
from pybrain.datasets import ClassificationDataSet
alldata = ClassificationDataSet(64,1, nb_classes=10)

f = open('digits.data', 'r')
for x in range(1, 3800):
#for x in range(1, 1500):
    line = f.readline()
    splits = line.split(',')
    result = splits[64]
    features = splits[:64];
    #print features,"count ",len(features),"result ",result
    alldata.addSample(features, result)

tstdata, trndata = alldata.splitWithProportion(0.25)

trndata._convertToOneOfMany()
tstdata._convertToOneOfMany()

#Build network with 20 neurons on each of 1 hidden layers
#Without hidden layer
print "in Dimension:",trndata.indim,"\nOut Dimension:",trndata.outdim
fnn = buildNetwork(trndata.indim, trndata.outdim, outclass=SoftmaxLayer)

#trainer = BackpropTrainer(fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)
trainer = BackpropTrainer(fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)
#trainer = BackpropTrainer(n, trndata)
#trainer.trainUntilConvergence()

#for i in range(20):
#Train network for 5 epochs
trainer.trainEpochs(50)

trnresult = percentError( trainer.testOnClassData(),
                              trndata['class'] )

tstresult = percentError( trainer.testOnClassData(
           dataset=tstdata ), tstdata['class'] )

print "epoch: %4d" % trainer.totalepochs, \
          "  train error: %5.2f%%" % trnresult, \
          "  test error: %5.2f%%" % tstresult
'''
print "Before training"
3
sq_err = []
for data in ds:
    input_entry = data[0]
    output_entry = data[1]
    pred_entry = n.activate(input_entry)
    print 'Actual:', output_entry, 'Predicted', pred_entry

    sq_err.append((pred_entry[0] - output_entry[0])**2)
print "RMSE: %.2f" % (sum(sq_err) / len(sq_err))


from pybrain.supervised.trainers import BackpropTrainer
trainer = BackpropTrainer(n, ds)
trainer.trainUntilConvergence()

print ""

print "After training"

sq_err = []
for data in ds:
    input_entry = data[0]
    output_entry = data[1]
    print 'Actual:', output_entry, 'Predicted', n.activate(input_entry)

    sq_err.append((pred_entry[0] - output_entry[0])**2)
print "RMSE: %.2f" % (sum(sq_err) / len(sq_err))
'''