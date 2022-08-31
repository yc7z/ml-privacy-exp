# ml-privacy-exp:

Various experiments for differentially private deep learning. We implemented the private SGD algorithm, which clips every per-sample gradient within a batch and adds Gaussian noises, thus achieving differential privacy guarantee. We included various classifiers and ran them on multiple datasets.

The experiments include running time and memory profilings, as well as accuracy results using adaptive optimization algorithms. Since differential privacy guarantee comes at the cost of increased memory usage and running time as well as degradation in testing accuracies, the main goal of these experiments was to provide insights into the reason behind the increased memory and runtime behind private SGD so they might be improved.

