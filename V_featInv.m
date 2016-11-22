% Usage: invert CNN features into images, same as deep-goggle
param_init

cnn_mode = 'caffe'; % 'caffe' or 'matconvnet'
gpu_id=0; % -1: cpu mode
mn='alex';
ln = 'c1';
ln = 'p5';


% 1. load feat, invert p5 feature
imName = 'data/images/test.jpg';
im_gt = U_prepare_image(single(imread(imName)),[],'alex',-2);

% 2. setup optimization param
U_optsInit
opts.feats = imName;
opts.task = 0;opts.objective = 'l2';
opts.dsp = 5;

% get the truncated model
if ~exist('net','var');U_loadModel;end

% initial image
init_img = randn([net.im_sz net.im_sz 3],'single') ;
load('x0_sigma.mat', 'x0_sigma');
init_img = init_img / norm(init_img(:)) * x0_sigma ;
opts.init = init_img;
% debug
% opts.init = im_gt;



opts= U_param(opts,mn,ln); % hard-coded learning rate and MRF energy parameters

% 3. do inversion
exp = experiment_init(mn, [], '', 'cnn', opts) ;
res = experiment_run(exp,net);
out = unit8(res.output{end});
