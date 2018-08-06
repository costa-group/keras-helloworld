# keras-helloworld
Dummy use of the keras lib.

# Installation
Keras works with different ML libs, I've choosed `TensorFlow`.
There are some errors on Ubuntu 16.04 with certain versions of `TensorFlow`, be sure of select the correct versions of the libraries.
I prefer to use `python 3` but it works also with `python 2.7`.

```sh
# Install python dev
sudo apt-get install python3 python3-dev python3-pip

# Install dependencies:
sudo -H pip3 install -U scipy numpy==1.14.5

# Install tensorflow
sudo -H pip3 install -U tensorflow==1.5

# Install keras
sudo -H pip3 install -U keras
```

For different architectures please read the following links:
- [TensorFlow installation](https://www.tensorflow.org/install/)
- [Keras installation](https://keras.io/#installation)

Check the installation with:
```sh
python -c "import tensorflow as tf; print(tf.__version__)"
```
