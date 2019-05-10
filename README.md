# KMNIST-SMAC
Used SMAC in PyTorch to optimize the hyperparameters of a CNN which is trained on KMNIST dataset.<br>
The optimizer used is Adam and the hyperparameters chosen for SMAC configuration space are <b>learning rate</b> and <b>beta1</b>.

# Runnning the code
First run `python3 download_data.py` to download the KMNIST dataset.
Then run `python3 kmnist_smac.py` to run the SMAC optimization.

## SMAC3
The documentation for SMAC3 can be found [here](https://automl.github.io/SMAC3/master)
