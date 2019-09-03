# Training Hub

## Prerequisites

- Programming language: [Python 3.7+](https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh)
- IDE: [PyCharm](https://www.jetbrains.com/pycharm/) or [Visual Studio Code](https://code.visualstudio.com/)
  - Plugins: [Kite](https://kite.com/), [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) (only for VSCode)
- Libraries & Frameworks: [Numpy](https://www.numpy.org/), [PyTorch](https://pytorch.org/), [OpenCV](https://opencv.org/), [TensorFlow](https://www.tensorflow.org/)
- Document: [LaTeX](https://www.latex-project.org/) (Highly recommended)
  - Tutorials: [Learn LaTeX in 30 minutes](https://www.overleaf.com/learn)
- Hardware: [NVIDIA GTX 1070+](https://www.nvidia.com/en-in/geforce/products/10series/geforce-gtx-1070/) (Optional)

## Syllabus

- Mathematics knowledge
- Python programming

## Math Knowledge

1. Probability and statistics

- [Standard deviation](http://www.mathsisfun.com/data/standard-deviation.html)

- [Conditional probability](https://www.khanacademy.org/math/statistics-probability/probability-library/conditional-probability-independence/v/calculating-conditional-probability)

2. Calculus

- [Derivative](https://www.khanacademy.org/math/calculus-home/taking-derivatives-calc)

- [Gradient](https://betterexplained.com/articles/vector-calculus-understanding-the-gradient/)

- [Chain rule](https://www.khanacademy.org/math/ap-calculus-ab/ab-derivative-rules/ab-chain-rule/a/chain-rule-review)

- [Partial derivative](https://www.mathsisfun.com/calculus/derivatives-partial.html)

- [Function transformation review](https://www.mathsisfun.com/sets/function-transformations.html)

3. Linear algebra:

- [Matrix review](https://www.mathsisfun.com/algebra/matrix-introduction.html)

- [Matrices and vectors](https://www.coursera.org/learn/machine-learning/lecture/38jIT/matrices-and-vectors)

- [Invertibiity](https://www.mathsisfun.com/algebra/matrix-inverse.html)

## Installation guide

```sh
# Designed for Ubuntu
bash ./install.sh

# For Ubuntu PC with NVIDIA GPU
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
conda install tensorflow-gpu

# For Ubuntu PC without NVIDIA GPU
conda install pytorch torchvision cpuonly -c pytorch
conda install tensorflow
```

**Installation For MacOS**:

- [Anaconda](https://www.anaconda.com/distribution/)
- [MacTeX 2019](http://www.tug.org/mactex/mactex-download.html)
- [TeXstudio](https://www.texstudio.org/)
- [PyTorch](https://pytorch.org/get-started/locally/): `conda install pytorch torchvision -c pytorch`
