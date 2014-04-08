from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer

from pybrain.structure import FeedForwardNetwork
n = FeedForwardNetwork()
from pybrain.structure import LinearLayer, SigmoidLayer, SoftmaxLayer

inLayers = range(64)
hiddenLayer_a = range(16)
#hiddenLayer_b = range(4)

for i in range(64):
    inLayers[i] = LinearLayer(1) 

for i in range(16):
    hiddenLayer_a[i] = SigmoidLayer(20)   
'''
for i in range(4):
    hiddenLayer_b[i] = SigmoidLayer(1)
'''
outLayer = LinearLayer(10)

for i in range(64):
    n.addInputModule(inLayers[i])

for i in range(16):
    n.addModule(hiddenLayer_a[i])

hiddenLayer_c = SigmoidLayer(100)
n.addModule(hiddenLayer_c)

'''
for i in range(4):
    n.addModule(hiddenLayer_b[i])
'''

n.addOutputModule(outLayer)


from pybrain.structure import FullConnection
for i in range(4):
    for j in range(4):
        print (j*2+i*16),(i*4+j)
        n.addConnection(FullConnection(inLayers[j*2+i*16], hiddenLayer_a[i*4+j]))
        n.addConnection(FullConnection(inLayers[j*2+i*16+1], hiddenLayer_a[i*4+j]))
        n.addConnection(FullConnection(inLayers[j*2+i*16+8], hiddenLayer_a[i*4+j]))
        n.addConnection(FullConnection(inLayers[j*2+i*16+9], hiddenLayer_a[i*4+j]))

for i in range(16):
    n.addConnection(FullConnection(hiddenLayer_a[i], hiddenLayer_c))

n.addConnection(FullConnection(hiddenLayer_c, outLayer))

'''
for i in range(2):
    for j in range(2):
        n.addConnection(FullConnection(hiddenLayer_a[j*2+i*8], hiddenLayer_b[i*2+j]))
        n.addConnection(FullConnection(hiddenLayer_a[j*2+i*8+1], hiddenLayer_b[i*2+j]))
        n.addConnection(FullConnection(hiddenLayer_a[j*2+i*8+4], hiddenLayer_b[i*2+j]))
        n.addConnection(FullConnection(hiddenLayer_a[j*2+i*8+5], hiddenLayer_b[i*2+j]))

for i in range(4):
    n.addConnection(FullConnection(hiddenLayer_b[i], outLayer))
'''
n.sortModules()

#from pybrain.datasets import SupervisedDataSet
#alldata = SupervisedDataSet(64, 1)
#alldata.addSample((0, 0, 0, 0), (0,))
from pybrain.datasets import ClassificationDataSet
alldata = ClassificationDataSet(64,1, nb_classes=10)

f = open('digits.data', 'r')
#for x in range(1, 3800):
#for x in range(1, 1500):
for line in f:
    #line = f.readline()
    splits = line.split(',')
    result = splits[64]
    features = splits[:64];
    #print features,"count ",len(features),"result ",result
    alldata.addSample(features, result)

tstdata, trndata = alldata.splitWithProportion(0.25)

trndata._convertToOneOfMany()
tstdata._convertToOneOfMany()

#Build network with 20 neurons on each of 1 hidden layers
#fnn = buildNetwork(trndata.indim, 20, trndata.outdim, outclass=SoftmaxLayer)
#Without hidden layer
#fnn = buildNetwork(trndata.indim, 64,32,trndata.outdim, outclass=SoftmaxLayer)
#fnn = buildNetwork(trndata.indim, 50,trndata.outdim, outclass=SoftmaxLayer)
#trainer = BackpropTrainer(fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)
trainer = BackpropTrainer(n, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)
#trainer = BackpropTrainer(n, trndata)
#trainer.trainUntilConvergence()

#Train network for 5 epochs
trainer.trainEpochs(15)

trnresult = percentError( trainer.testOnClassData(),
                              trndata['class'] )

tstresult = percentError( trainer.testOnClassData(
           dataset=tstdata ), tstdata['class'] )

print "epoch: %4d" % trainer.totalepochs, \
          "  train error: %5.2f%%" % trnresult, \
          "  test error: %5.2f%%" % tstresult