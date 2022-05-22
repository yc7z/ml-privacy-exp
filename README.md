# ml-privacy-exp:
Runtime experiments for differentially private deep learning. We implemented the private SGD algorithm, which clips every per-sample gradient within a batch and adds Gaussian noises, thus achieving differential privacy guarantee. The model used is LeNet and dataset is MNIST.

We are currently in the process of improving the runtime of this scheme, possibily by developing a new privacy mechanism.
