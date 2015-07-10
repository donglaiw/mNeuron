addpath('deep-goggle2')
addpath('util')

switch cnn_mode 
    case 'matconvnet'
        % matconvnet
        addpath(genpath('/data/vision/billf/donglai-lib/VisionLib/Donglai/DeepL/matconvnet_cpu'))
        if mid>2;error('matconvnet only support mid={1,2}, alexnet and vgg');end
    case 'caffe'
        % matcaffe
        addpath('/data/vision/torralba/gigaSUN/caffeCPU2/matlab/caffe');
        % load deployed caffe model
        layer=-1;caffe_init;
        % load truncated caffe model at layer lid
        layer=lid;caffe_init;
        layer_sz = size(response{lid});
        init_sz = size(response{1});
end 

% setup visualization parameters
cnn_init
opts.task = 1;opts.objective = 'oneclass';

% random initial image
stream=RandStream('mlfg6331_64');
stream.Substream=neuron_rand;
RandStream.setGlobalStream(stream)
init_img = randn(init_sz(1:3),'single') ;
% avg norm of images
load('x0_sigma.mat', 'x0_sigma');
init_img = init_img / norm(init_img(:)) * x0_sigma ;


