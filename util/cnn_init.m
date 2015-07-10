exp = {} ;
ver = 'results' ;
clear opts
opts.learningRate = 0.004 * [...
    ones(1,100), ...
    0.1 * ones(1,200), ...
    0.01 * ones(1,100)];
opts.objective = 'l2' ;
opts.beta = 2 ;
opts.lambdaL2 = 8e-10 ;
opts.lambdaTV = 1e0 ;
opts.TVbeta = 2;

% dw
opts.init = [];
opts.layer_ind = [];
opts.dsp = 25;
opts.do_dc = 0;
opts.feats = [];
opts.do_thres =0;
% 2. for one class bp
opts.mask = [];
opts.grad_ran = [0,0];
% 3. for inpainting
opts.xmask = [];
opts.freezeDropout = 1;
% for pool1 regu
opts.regu =[];
% patch prior
opts.pd=[];
% color normalize
opts.im_w=[];
% nabla_2
opts.lambdaTV2=0;
opts.TV2beta=2;


