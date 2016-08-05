# mNeuron: A Matlab Plugin to Visualize Neurons from Convolutional Neural Network (CNN)

## Features:
- Result: [**Visualization Page**](http://vision03.csail.mit.edu/cnn_art/index.html), [**report**](http://vision03.csail.mit.edu/cnn_art/data/cnn_visual_arxiv.pdf)
- Support models from [**caffe (June 2016)**](https://github.com/BVLC/caffe) and [**matconvnet (1.0-beta12, May 2015)**](http://www.vlfeat.org/matconvnet/)
(caffe support: AlexNet/VGG-16/NIN/GoogleNet; matconvet support: AlexNet)
(matconvnet: upgrade the models with the `vl_simplenn_tidy` function.)

## Demos:
1. Visualize single neurons with caffe model:
  1. Edit **param_init.m**: caffe/matcaffe location
  2. Download imagenet models
     [**link**](http://vision03.csail.mit.edu/cnn_art/models/) and put under models/
  3. Edit and run: **V_neuron_single.m**

## Under Construction
1. Visualize Intra-class variation
2. Visualize hierarchical binary CNN code
3. Image Completion

## Update
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
