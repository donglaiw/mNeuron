% Usage: invert neurons from model 'mid' at certain layer 'lid'
% caffe supoort: alexnet, vgg, nin, googlenet
% matconvnet supoort: alexnet, vgg

% Things to play with
% a) select model/layer/neuron/rf: 1.1, 1.2
% b) optimization parameter: 2.3

% to reload model: clear net;V_neuron_single;

param_init
% --------- USER INPUT -----------
%% 1.1 select and load model
cnn_mode = 'caffe'; % 'caffe' or 'matconvnet'
%cnn_mode = 'matconvnet';
gpu_id=0; % -1: cpu mode
mn='alex';ln = 'p5';
% details in util/U_loadModel.m
% model names (mn): 
%   - alex, vgg16, nin, gnet
% layer names (ln):
%   - convolution layers: c1,..,c5
%   - pooling layers: p5
%   - fully-connected layers: f6,..,f8
%   - cccp layers (NIN): cp6,..,cp8
%   - inception layers (GoogleNet): i4a

%% 1.2 select which neuron and number of tiles 
neuron_id=10; % neuron channel id
neuron_rf=6; % neuron_rf x neuron_rf tiles will be visualized (for alexnet, range: 1-6)
neuron_rand=10; % random seed for initializtion
% file of initial image, random if init_file=[];
init_file = [];


% --------- READY TO RUN -----------
% setup visualization parameters
U_optsInit
opts.task = 1;opts.objective = 'oneclass';
% display every opts.dsp iterations (no display if negative)
opts.dsp = 5;
% get the truncated model
if ~exist('net','var');U_loadModel;end


% random initial image
rng(neuron_rand);
init_img = randn([net.im_sz net.im_sz 3],'single') ;
% avg norm of images
load('x0_sigma.mat', 'x0_sigma');
init_img = init_img / norm(init_img(:)) * x0_sigma ;


% 2. visualize model at certain layer
% 2.1 objective function: optimize the sum of the masked neuron response
p5_m1 = zeros(net.layer_sz(1:3),'single');
neuron_rf = min(neuron_rf,net.layer_sz(1));
ind_st = max(0,ceil((net.layer_sz(1)-neuron_rf)/2));
p5_m1(ind_st+(1:neuron_rf),ind_st+(1:neuron_rf),neuron_id)=1;
opts.mask = p5_m1;

% 2.2 initial image
if exist([init_file],'file')
    opts.init = U_prepare_image(single(imread(init_file)),'',mn,-2);
else
    % random initialization
    opts.init = init_img;
end

% 2.3 optization parameters
opts= U_param(opts,mn,ln); % hard-coded learning rate and MRF energy parameters

% 3. run experiment
exp = experiment_init(mn,[], '', 'cnn', opts) ;
res = experiment_run(exp,net);
out=uint8([res.output{end}]);

%% output
%output_folder = sprintf('result/%s_%s/',mn,ln); mkdir(output_folder)
%output_name = sprintf('%d_%d_%d.png',neuron_id,neuron_rf,neuron_rand);
% imwrite(out,[output_folder output_name])
% e.g. use bigger rf to initialize smaller rf 
%if ~exist([init_file],'file');imwrite(out,init_file);end
