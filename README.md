# Algorithm 1


It will require "ML LIBRARY"???? (TO DO).

The folder algorithm1 has the algorithm that we are developing ready
to be integrated on iRankFinder, but to simplify things I have create
a script called "main" to reduce the dependencies.

We only need: `pplpy` (I couldn't remove this), `pyLPi`, `pyParser`, `z3` and `argparser` (and also the
ML Lib) 

## Installation

Download the script
[install_externals.sh](https://raw.githubusercontent.com/costa-group/iRankFinder/master/installer/install_externals.sh),
run it... then run:

```sh
sudo -H pip3 install -U "git+https://github.com/jesusjda/pplpy.git#egg=pplpy" --process-dependency-links
sudo -H pip3 install -U "git+https://github.com/jesusjda/pyLPi.git#egg=pyLPi" --process-dependency-links
sudo -H pip3 install -U "git+https://github.com/jesusjda/pyParser.git#egg=genericparser" --process-dependency-links
```

to run the tool:


```sh
python3 main.py -f [FILES] -v [VERBOSITY LEVEL]
```

-------------
-------------


# keras-helloworld
Dummy use of the [Keras] lib.

## Installation
[Keras] works with different ML libs, I've chosen [TensorFlow].
There are some errors on Ubuntu 16.04 with certain versions of [TensorFlow], be sure of select the correct versions of the libraries.
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

[//]: # (Links used in the body)

   [TensorFlow]: <https://tensorflow.org>
   [Keras]: <https://keras.io>
