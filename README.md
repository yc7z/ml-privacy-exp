# ml-privacy-exp:

## Descriptions

Various experiments for differentially private deep learning. We implemented the private SGD algorithm, which clips every per-sample gradient within a batch and adds Gaussian noises, thus achieving differential privacy guarantee. We included various classifiers and ran them on multiple datasets.

The experiments include running time and memory profilings, as well as accuracy results using adaptive optimization algorithms and gradient compression. Since differential privacy guarantee comes at the cost of increased memory usage and running time as well as degradation in testing accuracies, the main goal of these experiments was to provide insights into the reason behind the increased memory and runtime behind private SGD so they might be improved.

We included implementations of the private SGD algorithm in OPACUS as well as the Functorch library that comes with PyTorch 1.12.

## Requirements
Install PyTorch, Torchvision, Opacus, and TensorFlow Privacy.

## Usage
A summary of the various scripts:
* ```dp_timing.py```: measures the absolute running time of private training vs public training.
* ```dp_runtime_profile.py```: gives the Tensorboard profiling results of private training.
* ```dp_memory.py```: provides the memory usage information of private training.
* ```naive_momentum.py```: contains experiments of various adaptive optimization algorithms and topk compression applied to private training.
* ```dp_pub_compress.py```: contains experiments of utilizing public information to inform topk compression.
* ```utils_plus.py``` and ```utils.py``` contains implementations of the privacy mechanism.
* ```models.py``` contains the models used in our experiments. One can extends this file if one wishes to run experiments on other model architectures.


We implemented DP-SGD is implemented in both OPACUS and Functorch. You can toggle the flag --mode to choose which version to run:
* ```--mode functorch_dp``` to run the functorch implementation.
* ```--mode opacus``` to run the Opacus implementation.
* ```--mode public``` to run public training without differential privacy.


