from pybrain.structure import FeedForwardNetwork
n = FeedForwardNetwork()
from pybrain.structure import LinearLayer, SigmoidLayer, SoftmaxLayer
inLayer1 = LinearLayer(2)
hiddenLayer1 = SigmoidLayer(3)
outLayer = LinearLayer(1)
inLayer2 = LinearLayer(2)
hiddenLayer2 = SigmoidLayer(3)
n.addInputModule(inLayer1)
n.addInputModule(inLayer2)
n.addModule(hiddenLayer1)
n.addModule(hiddenLayer2)
n.addOutputModule(outLayer)
from pybrain.structure import FullConnection
n.addConnection(FullConnection(inLayer1, hiddenLayer1))
n.addConnection(FullConnection(inLayer2, hiddenLayer2))
n.addConnection(FullConnection(hiddenLayer1, outLayer))
n.addConnection(FullConnection(hiddenLayer2, outLayer))
n.sortModules()

from pybrain.datasets import SupervisedDataSet
ds = SupervisedDataSet(4, 1)


ds.addSample((0, 0, 0, 0), (0,))
ds.addSample((0, 0, 1, 1), (1,))
ds.addSample((0, 0, 0, 1), (0,))
ds.addSample((0, 1, 1, 0), (1,))

print "Before training"

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
