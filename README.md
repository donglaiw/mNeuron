# CNN Visualization toolbox 
(support caffe and matconvnet)
 
For Caffe, we have four models:
- mid=1: AlexNet
- mid=2: VGG-16 
- mid=3: NIN 
- mid=4: GoogleNet 

## Progress:
1. invert pool5 neuron with caffe model:
  1. Edit matcaffe location in **param_init.m**
  2. Edit alexnet.caffemodel location in **util/caffe_init.m**
  3. Run: **V_neuron.m**


