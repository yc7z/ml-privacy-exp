# ml-privacy-exp:

## Descriptions

Various experiments for differentially private deep learning. We implemented the private SGD algorithm, which clips every per-sample gradient within a batch and adds Gaussian noises, thus achieving differential privacy guarantee. We included various classifiers and ran them on multiple datasets.

The experiments include running time and memory profilings, as well as accuracy results using adaptive optimization algorithms and gradient compression. Since differential privacy guarantee comes at the cost of increased memory usage and running time as well as degradation in testing accuracies, the main goal of these experiments was to provide insights into the reason behind the increased memory and runtime behind private SGD so they might be improved.

We included implementations of the private SGD algorithm in OPACUS as well as the Functorch library that comes with PyTorch 1.12.

## Requirements
Install PyTorch, Torchvision, Opacus, and TensorFlow Privacy.

## Usage

For each of the scripts mentioned below, DP-SGD is implemented in both OPACUS and Functorch. You can toggle the flag --mode to choose which version to run:
* --mode functorch_dp to run the functorch implementation.
* --mode opacus to run the Opacus implementation.
* --mode public to run public training without differential privacy.



