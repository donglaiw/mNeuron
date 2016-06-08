# mNeuron: A Matlab Plugin to Visualize Neurons from Convolutional Neural Network (CNN)
(support )

## Features:
1. Support models from [**caffe (June 2016)**](https://github.com/BVLC/caffe) and [**matconvnet (1.0-beta12, May 2015)**](http://www.vlfeat.org/matconvnet/)
2. Result: [**Visualization Page**](http://vision03.csail.mit.edu/cnn_art/index.html)
3. Compatible version of CNN models [download link](http://vision03.csail.mit.edu/manip/HITs_vimeo/mNeuron/)

## Demos:
(caffe support: AlexNet/VGG-16/NIN/GoogleNet; matconvet support: AlexNet/VGG-16)

1. Visualize single neurons with caffe model:
  1. Edit matcaffe location in **param_init.m**
  2. Edit alexnet.caffemodel location in **util/caffe_init.m**
  3. Run: **V_neuron_single.m**

1. Image Completion
  1. Run **V_app_inpaint.m** 

## Under Construction
1. Visualize Intra-class variation
2. Visualize hierarchical binary CNN code

## Update
+ 2016.06: code organization
    ..* update caffe version to June 2016
    ..* add links to compatible CNN models
+ 2016.04: add object insertion
    ..* object insertion (V_app_inpaint.m)
+ 2015.07: initial release
    ..* support caffe (July 2015) and matconvnet (1.0-beta12, May 2015)
    ..* single neuron visualization (V_neuron_single.m)

## Acknowledgement
The code is heavily based on aravindhm's [**deep-goggle**](https://github.com/aravindhm/deep-goggle)
