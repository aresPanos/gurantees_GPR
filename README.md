# Mercer Gaussian Process (MGP) and Fourier Gaussian Process (FGP) Regression #

We provide the code used in our paper "**How Good are Low-Rank Approximations in Gaussian Process Regression?**" to run experiments on the real-world datasets. The code includes implementation of Mercer GP (using dimensionality reduction) and Fourier GP. We also include the GPFlow code to run SGPR model.

## Requirements ##
TensorFlow - version 2.1.0  
TensorFlow Probability - version 0.9.0  
GPflow - version 2.0.0 or newer  
silence-tensorflow - version 1.1.1 (optional)

## Flags ##
* batch_size: Batch size for MGP (due to the included shallow neural network) (integer - default=2048)
* num_epochs: Display loss function value every FLAGS.display_freq epochs (integer - default=100)
* num_splits: Number of random data splits used - number of experiments run for a model (integer - default=1)
* display_freq: Display loss function value every *display_freq epochs* (integer - default=10)
* rank: Rank r for MGP, FGP, SGPR (integer - default=10)
* d_mgp: Number of output dimensions for MGP\'s projection (integer - default=5)
* dataset: Dataset name (string - available names=[elevators, protein, sarcos, 3droad] - default=elevators)

## Source code ##

The following files can be found in the **src** directory :  

- *models.py*: implementation of MGP  and FGP
- *helper.py*: various utility functions
- *hermite_coeff.npy*: a numpy array containing the Hermite polynomial coefficients needed for the DMGP model
- *run_experiments.py*: code for running models MGP, FGP, and SGPR on the real-world datasets used in the paper

## Examples ##
You can run the code with the configuration of your choice using the following command

```
# Train MGP, FGP, SGPR models over the Protein dataset and repeat experiments 5 times
# Set the number of epochs equal to 500 
# Print the values of the log-marginal likelihood every 5 epochs.
# The rank of the kernel approximation is chosen to be 50

python src/run_experiments.py --dataset=protein --display_freq=5 --num_splits=5 --rank=50

```


