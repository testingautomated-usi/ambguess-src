# True Ambiguity - Paper: Reproduction Package
This is the replication package for our paper "Generating and Detecting True Ambiguity: A Forgotten Danger in DNN Supervision Testing", accepted at EMSE.

A preprint is available on [arXiv](https://arxiv.org/abs/2207.10495).

## Access to precompiled datasets
If you are not interested in our sources, but just in the released datasets, you can find them here:
- [MNIST-Ambiguous on Huggingface Datasets](https://huggingface.co/datasets/mweiss/mnist_ambiguous)
- [Fashion-Mnist-Ambiguous on Huggingface Datasets](https://huggingface.co/datasets/mweiss/fashion_mnist_ambiguous)
- [testingautomated-usi/ambiguous-datasets repo](https://github.com/testingautomated-usi/ambiguous-datasets) (serialized numpy arrays attached to github release)

## Dependencies
On your machine, you'll need the following requirements:
- Docker
- The uncompressed ambiguess-artifacts folder, further referred to as `/path/to/artifacts/`
- If running on linux with an nvidia-gpu, install the [nvidia-docker toolkit](https://github.com/NVIDIA/nvidia-docker)
  which will allow you to use a GPU for training and inference.

## Step 0: Building docker container
Navigate into the ambiguess repository and run the following command:
> docker build -t ambiguess:snapshot .

## Step 1: Running the container
Start the container with the following command (replacing `/path/to/artifacts/` with the path to the artifacts folder):
> docker run -it --rm -v /path/to/artifacts/:/artifacts -w /ambiguess ambiguess:snapshot

Note: 
- If you are using nvidia-docker, add `--gpus all` after the `--rm` flag.
- You can find our `/path/to/artifacts/` (thus our model weights, ...) on zenodo.

You should now see a Tensorflow welcome message.

## Step 2: Running the reproduction package CLI

You can reproduce the results of the paper by using our provided command line interface as follows:

> python cli.py COMMAND [ARGS]

- Run `python cli.py --help` for more information on the available commands.
- Run `python cli.py COMMAND --help` for more information on the available arguments for a specific command
(e.g. `python cli.py train --help`).

**Attention: Running any of these commands will modify the contents of the `/path/to/artifacts/` folder.**

You can exit the docker container by entering `exit`.
