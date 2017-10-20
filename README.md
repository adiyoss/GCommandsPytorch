# ConvNets for Speech Commands Recognition

Training ConvNet models using [Google Commands Dataset](https://research.googleblog.com/2017/08/launching-speech-commands-dataset.html), implemented in [PyTorch](http://pytorch.org).
<!-- This repo contains data loader for the [Google Commands Dataset](https://research.googleblog.com/2017/08/launching-speech-commands-dataset.html), as well as training scripts for several ConvNet models, written in [PyTorch](http://pytorch.org). -->

## Features
* Training and testing ConvNets.
* Arrange [Google Commands Dataset](https://research.googleblog.com/2017/08/launching-speech-commands-dataset.html), in an Train, Test, Valid folders for easy loading.
* Dataset loader.

## Installation
Several libraries are needed to be installed in order to extract spectrograms and train the models.

* Install [PyTorch](https://github.com/pytorch/pytorch#installation) if you haven't already :)
* Install [LibRosa](https://github.com/librosa/librosa)

## Usage

### Google Commands Dataset
Download and extract the [Google Commands Dataset](https://research.googleblog.com/2017/08/launching-speech-commands-dataset.html).

To make the arrange the data run the following command:
```
python make_dataset.py <google-command-folder> --out_path <path to save the data the new format>
```

### Custom Dataset
You can also use the data loader and training scripts for your own custom dataset.
In order to do so the dataset should be arrange in the following way:
```
root/up/kazabobo.wav
root/up/asdkojv.wav
root/up/lasdsa.wav
root/right/blabla.wav
root/right/nsdf3.wav
root/right/asd932.wav
```

### Training
Use `python run.py --help` for more parameters and options.

```
python run.py --train_path <train_data_path> --valid_path <valid_data_path> --test_path <test_data_path>
```

### Results
Accuracy results for the train, validation and test sets using two ConvNet models (LeNet5 and VGG11). 

In order to reproduce the below results just exec the run.py file with default parameters.
Results may be improved using deeper models (VGG13, VGG19), or better hyper-parameters optimization.

| Model | Train acc. | Valid acc. | Test acc.|
| ------------- | ------------- | ------------- | ------------- |
| LeNet5  | 99% (50742/51088)  | 90% (6093/6798) | 89% (6096/6835) | 
| VGG11  |  97% (49793/51088) | 94% (6361/6798) | 94% (6432/6835) |

