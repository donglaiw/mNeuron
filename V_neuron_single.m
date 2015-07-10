% invert neurons from model 'mid' at certain layer 'lid'
% modified from deep-goggle/

%% 1. model parameters
% caffe supoort: alexnet, vgg, nin, googlenet
% matconvnet supoort: alexnet, vgg
cnn_mode='caffe'; % 'caffe' or 'matconvnet'
%cnn_mode='matconvnet'; % 'caffe' or 'matconvnet'

% visualize pool5 neurons for each model
% mid=1; lid=10; % alexnet
% mid=2; lid=19; % VGG-16
mid=3; lid=13; % NIN
% mid=4; lid=45; % GoogleNet
% visualize fc8 neurons for each model
% mid=1; lid=14; % alexnet
% mid=2; lid=19; % VGG-16
% mid=3; lid=13; % NIN
% mid=4; lid=45; % GoogleNet


gpu_id=1; % -1: cpu mode
% neuron_id=1; % neuron channel id
% neuron_rf=14; % receptive field size
neuron_rand=1; % random seed for initializtion

output_folder = sprintf('result/%d_%d/',mid,lid); mkdir(output_folder)
output_name = sprintf('%d_%d_%d.png',neuron_id,neuron_rf,neuron_rand);
init_file = [output_folder sprintf('init_%d.png',neuron_id)];

%% for debug 
% display every opts.dsp iterations (no display if negative)
opts.dsp = 5;
% to avoid reloading models: do_setup=0;V_neuron_single;
if ~exist('do_setup','var');do_setup=1;end

% 1. load model
if do_setup;V_neuron_setup;end
%return


% 2. visualize model at certain layer
% 2.1 objective function: optimize the sum of the masked neuron response
p5_m1 = zeros(layer_sz(1:3),'single');
neuron_rf = min(neuron_rf,layer_sz(1));
ind_st = max(0,ceil((layer_sz(1)-neuron_rf)/2));
p5_m1(ind_st+(1:neuron_rf),ind_st+(1:neuron_rf),neuron_id)=1;
opts.mask = p5_m1;

% 2.2 initial image
if exist([init_file],'file')
    opts.init = U_prepare_image(single(imread(init_file)),'',mid,-2);
else
    % random initialization
    opts.init = init_img;
end

% 2.3 optization parameters
opts= V_neuron_param(1,mid,lid,opts); % hard-coded learning rate and MRF energy parameters

% 3. run experiment
exp = experiment_init(mid, lid, [], '', 'cnn', opts) ;
res = experiment_run(exp);
out=uint8([res.output{end}]);

imwrite(out,[output_folder output_name])
% e.g. use bigger rf to initialize smaller rf 
if ~exist([init_file],'file')
    imwrite(out,init_file)
end
