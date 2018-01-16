# mNeuron: A Matlab Plugin to Visualize Neurons from Convolutional Neural Network (CNN)

## Features:
- Result: [**Visualization Page**](http://vision03.csail.mit.edu/cnn_art/index.html), [**arxiv**](http://vision03.csail.mit.edu/cnn_art/data/cnn_visual_arxiv.pdf)
- Support models from [**caffe (June 2016)**](https://github.com/BVLC/caffe) and [**matconvnet (1.0-beta12, May 2015)**](http://www.vlfeat.org/matconvnet/)

(caffe support: AlexNet/VGG-16/NIN/GoogleNet; matconvet support: AlexNet)

## Download Pre-train Models:
- [**link**](http://vision03.csail.mit.edu/cnn_art/models/) and put under models/
- Edit **param_init.m**: caffe/matcaffe location

## Demos:
1. Visualize single neurons from CNN model:
  1. Edit and run: **V_neuronInv.m**

2. Image Completion
  1. Download object topics
     [**link**](http://vision03.csail.mit.edu/cnn_art/fc7-topic.zip) and put under data/fc7-topic/
  2. Edit and run: **V_inpaint.m**

3. Feature Inversion, same as [**deep-goggle**](https://github.com/aravindhm/deep-goggle)
  1. Edit and run: **V_featInv.m**

## Under Construction
1. Visualize Intra-class variation
2. Visualize hierarchical binary CNN code

## Update
- 2016.11: bug fixed
- 2016.08: code organization
    - update caffe version to June 2016
    - add links to compatible CNN models
- 2016.04: add object insertion
    - object insertion (V_app_inpaint.m)
- 2015.07: initial release
    - support caffe (July 2015) and matconvnet (1.0-beta12, May 2015)
    - single neuron visualization (V_neuron_single.m)

## Acknowledgement
The code is heavily based on aravindhm's [**deep-goggle**](https://github.com/aravindhm/deep-goggle)
