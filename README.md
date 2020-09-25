# Adversarial Knowledge Transfer from Unlabeled Data 

This repository is the official implementation our paper titled "[Adversarial Knowledge Transfer from Unlabeled Data](https://arxiv.org/abs/2008.05746)" accepted to [ACM-MM 2020](https://2020.acmmm.org/).


## Implementation Details

Our implementation is in PyTorch [1] with python 3.6.7. We train all our 
model on GeForce RTX 2080 Ti GPUs. This implementation currently uses
one gpu and can be modified to use multiple gpus for larger batch size.


## Python Packages

Please refer to the requirements.txt file for all the packages we
used to create the environment for training our models. We create an
environment in anaconda.


## Datasets
This is a working code of our proposed method for PASCAL-VOC/ImageNet 
experiment. We use PASCAL-VOC[2] dataset as the labeled target dataset 
and ImageNet[3] as unlabeled source dataset.


## Usage

### To train a model on PASCAL-VOC and ImageNet experiment WITH GPU
    python train.py --pascal_path <path-to-pascal-voc-dataset> \
                    --imgnet_path <path-to-imagenet-dataset>   \
                    --gpu <gpu-id-to-use>

    python train.py --pascal_path /datasets/pascal-voc-2007/   \
                    --imgnet_path /datasets/imagenet-dataset/   \
                    --gpu 0



## To test your trained model WITH GPU
    python train.py --pascal_path <path-to-pascal-voc-dataset> \
                    --model <path-to-trained-model>            \
                    --gpu <gpu-id-to-use>                      \
                    --test 1

    python train.py --pascal_path /datasets/pascal-voc-2007/   \
                    --model ./checkpoints/best-model.pth       \
                    --gpu 0                                    \
                    --test 1

### To train a model on PASCAL-VOC and ImageNet experiment WITHOUT GPU
    python train.py --pascal_path <path-to-pascal-voc-dataset> \
                    --imgnet_path <path-to-imagenet-dataset>                      

    python train.py --pascal_path /datasets/pascal-voc-2007/   \
                    --imgnet_path /datasets/imagenet-dataset/   

### To test your trained model WITHOUT GPU
    python train.py --pascal_path <path-to-pascal-voc-dataset> \
                    --model <path-to-trained-model>            \
                    --test 1

    python train.py --pascal_path /datasets/pascal-voc-2007/   \
                    --model ./checkpoints/best-model.pth       \
                    --test 1

## Citation
Will be added soon.
    
## Contact
Please contact the first author Akash Gupta ([agupt013@ucr.edu](agupt013@ucr.edu)) for any questions.
    
## References

1. Paszke, Adam, Sam Gross, Soumith Chintala, Gregory Chanan, Edward
    Yang, Zachary DeVito, Zeming Lin, Alban Desmaison, Luca Antiga, 
    and Adam Lerer. "Automatic differentiation in pytorch." (2017).<br />
1. Everingham, Mark, Luc Van Gool, Christopher KI Williams, John Winn,
    and Andrew Zisserman. "The pascal visual object classes (voc) 
    challenge." International journal of computer vision 88, no. 2 
    (2010): 303-338.<br />
1. Deng, Jia, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li 
    Fei-Fei. "Imagenet: A large-scale hierarchical image database." 
    In 2009 IEEE conference on computer vision and pattern recognition, 
    pp. 248-255. Ieee, 2009.
    


