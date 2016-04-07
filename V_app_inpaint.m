addpath('util')
addpath('data')
% parameters to set
TOPIC_ID=1;
MODEL_ID='caffe-ref';
mid=1;lid=20; % fc7 
cnn_mode='matconvnet'; % 'caffe' or 'matconvnet'
neuron_rand=1;

init_sz = [227 227 3];
if ~exist('do_setup','var');do_setup=1;end
% 1. load model
if do_setup;V_neuron_setup;end

% cat example
doid=1;sm_id = 333;nn='cat';
cnn_init

opts.objective = 'oneclass';
opts.task = 1;
opts.layer= lid; 
opts.dsp = 5;
opts.do_thres=1;

load(['data/inpaint/locate_' nn],'mask_rm','im0','p5_mask') 
% choose a class
opts.mask=zeros(1,1000,'single');
opts.mask(sm_id)=1;

% choose a pool5
lid2=16;
tmp_p5 = imdilate(p5_mask,strel('disk',1));
%tmp_p5(3,3)=0; % create extra parts
opts.fmask = {{lid2,repmat(tmp_p5,[1,1,256])}};
% choose a topic
topic= load(sprintf('data/fc7-topic/%d.mat',sm_id-1));
opts.fmask={opts.fmask{:},{20,reshape(topic.W(:,TOPIC_ID),[1,1,4096])}};

% region to modify
% hand tune...
opts.xmask = single(zeros(init_sz));
bb=[45 150 60 180];
opts.xmask(bb(1):bb(2),bb(3):bb(4),:) = 1;
init_pb = single(imread(['data/inpaint/locate_' nn '.png'])); 

% initial image/db
%im_init = U_prepare_image(init_pb,[],1,1);
im_init = bsxfun(@minus,init_pb,reshape([103.939, 116.779, 123.68],[1,1,3]));
% pm-init 
r_std=20;
opts.init = U_init_r(im_init,opts.xmask,r_std,im_init);

opts.lambdaL2 = 1e-2 ;opts.lambdaTV = 3e4; opts.lambdaD = 1e0;
opts.grad_ran=[0 100];
opts.learningRate = 0.00004 * [...
0.05 * ones(1,70), ...
0.02 * ones(1,70), ...
0.02 * ones(1,70)];
opts.np=[];

% same order with the fc term
exp = experiment_init(MODEL_ID, opts.layer, [], '', 'cnn', opts) ;
res = experiment_run(exp);

im0 = uint8(res.output{end}); 
imwrite(im0,[DD_out sprintf('ip_%s_%d_%d_%d_%d.png',nn,tid,doid,TOPIC_ID,doid3)])
