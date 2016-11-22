% only work for matconvet now

param_init
% parameters to set
TOPIC_ID=2;
mn='alex';ln='f7'; % fc7 
cnn_mode='matconvnet'; % 'caffe' or 'matconvnet'
neuron_rand=1;

init_sz = [227 227 3];
% 1. load model

% cat example
% load model
opts.task = 2;opts.objective = 'oneclass';
if ~exist('net','var');U_loadModel;end
classId = 333;oldClassName='cat';

opts.dsp = 5;
opts.do_thres=1;


load(['data/inpaint/locate_' oldClassName],'mask_rm','im0','p5_mask') 
% choose a class
opts.mask=zeros(1,1000,'single');
opts.mask(classId)=1;


% choose a pool5
tmp_p5 = imdilate(p5_mask,strel('disk',1));
tmp_p5(3,3)=0;
%tmp_p5 = p5_mask;
opts.fmask = {};
% choose a fc7 topic
topic= load(sprintf('data/fc7-topic/%d.mat',classId-1));
opts.fmask={{16,repmat(tmp_p5,[1,1,256])},{20,reshape(topic.W(:,TOPIC_ID)>1,[1,1,4096])}};

% region to modify
% hand tune...
opts.xmask = single(zeros(init_sz));
bb=[45 150 60 180];
opts.xmask(bb(1):bb(2),bb(3):bb(4),:) = 1;
init_pb = single(imread(['data/inpaint/locate_' oldClassName '.png'])); 

% initial image/db
%im_init = U_prepare_image(init_pb,[],1,1);
im_init = bsxfun(@minus,init_pb,reshape([103.939, 116.779, 123.68],[1,1,3]));
% pm-init 
r_std=20;
opts.init = U_init_r(im_init,opts.xmask,r_std,im_init);

% 2.3 optization parameters
opts= U_param(opts,mn,ln); % hard-coded learning rate and MRF energy parameters

% same order with the fc term
exp = experiment_init(mn, [], '', 'cnn', opts) ;
res = experiment_run(exp,net);

im0 = uint8(res.output{end}); 
%imwrite(im0,[DD_out sprintf('ip_%s_%d_%d_%d_%d.png',nn,tid,doid,TOPIC_ID,doid3)])
