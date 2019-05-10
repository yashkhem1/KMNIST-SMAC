import torch
import torchvision
from kmnist_cnn import CNN, train
import numpy as np
import matplotlib.pyplot as plt

#--------------SMAC---------------------------#
# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import InCondition

# Import SMAC-utilities
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
#---------------------------------------------#

batch_size = 100
epochs = 12

torch.set_default_tensor_type('torch.DoubleTensor')
# img = np.load("kmnist-train-imgs/arr_0.npy")[0]
# plt.imshow(img)
# # plt.show()
# print(img.shape)
train_X = torch.from_numpy(np.load("kmnist-train-imgs.npz")["arr_0"]/255.0).unsqueeze(1)
train_Y = torch.from_numpy(np.load("kmnist-train-labels.npz")["arr_0"])
test_X = torch.from_numpy(np.load("kmnist-test-imgs.npz")["arr_0"]/255.0).unsqueeze(1)[0:5000]
test_Y = torch.from_numpy(np.load("kmnist-test-labels.npz")["arr_0"])[0:5000]
val_X = torch.from_numpy(np.load("kmnist-test-imgs.npz")["arr_0"]/255.0).unsqueeze(1)[5000:]
val_Y = torch.from_numpy(np.load("kmnist-test-labels.npz")["arr_0"])[5000:]

trainDataset = torch.utils.data.TensorDataset(train_X,train_Y)
testDataset = torch.utils.data.TensorDataset(test_X,test_Y)
valDataset = torch.utils.data.TensorDataset(val_X,val_Y)

trainDataloader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
testDataloader = torch.utils.data.DataLoader(testDataset, batch_size=batch_size, shuffle=True)
valDataloader = torch.utils.data.DataLoader(valDataset, batch_size=batch_size, shuffle=True)

# net = CNN()

# train(net,0.001,0.9,trainDataloader, valDataloader,epochs)

##-----SMAC---------##
cs = ConfigurationSpace()

lr = UniformFloatHyperparameter("lr",0.0001,0.01,default_value=0.001)
beta1 = UniformFloatHyperparameter("beta1",0.5,0.99,default_value=0.9)
cs.add_hyperparameters([lr,beta1])

def kmnist_from_cfg(cfg):
    cfg = {k : cfg[k] for k in cfg if cfg[k]}
    lr = cfg["lr"]
    beta1 = cfg["beta1"]
    model = CNN()
    val_accuracy = train(model, lr, beta1, trainDataloader, valDataloader, epochs)
    return 1 - val_accuracy  # Minimize


# Scenario object
scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
                     "runcount-limit": 200,  # maximum function evaluations
                     "cs": cs,               # configuration space
                     "deterministic": "true"
                     })

# Optimize, using a SMAC-object
print("Optimizing! Depending on your machine, this might take a few minutes.")
smac = SMAC(scenario=scenario, rng=np.random.RandomState(42),
            tae_runner=kmnist_from_cfg)
smac.solver.intensifier.tae_runner.use_pynisher = False

incumbent = smac.optimize()


inc_value = kmnist_from_cfg(incumbent)

print("Optimized Value: %.2f" % (inc_value))

print("Optimum Parameters: lr = ", incumbent["lr"], " beta1 = ", incumbent["beta1"])
