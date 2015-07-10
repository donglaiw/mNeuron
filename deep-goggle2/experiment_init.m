function exp = experiment_init(model, layer, imageName, prefix, suffix, varargin)
% Initialize an experiment
% from deep-goggle


if nargin < 4
    prefix = '' ;
end

if nargin < 6
    suffix = '' ;
end

if ~isempty(imageName)
    [imageDir,imageName,imageExt] = fileparts(imageName) ;
else
    imageDir=[];imageExt=[];
end
if isempty(imageDir), imageDir = 'data/images' ; end

exp.expDir = fullfile('data', prefix, suffix) ;
exp.model = model ;
exp.layer = layer ;
exp.name = imageName ;
exp.useHoggle = false ;
exp.path = fullfile(imageDir, [imageName, imageExt]) ;
exp.opts.dropout = 0 ;
exp.opts.neigh = +inf ;
exp.opts.filterGroup = NaN ;
exp.opts.objective = 'l2' ;
exp.opts.learningRate = 0.1 * ones(1,100) ;
exp.opts.maxNumIterations = +inf ;
exp.opts.beta = 2 ;
exp.opts.lambdaTV = 100 ;
exp.opts.lambdaL2 = 0.1 ;
exp.opts.TVbeta = 1;
exp.opts.numRepeats = 1 ;
exp.opts.optim_method = 'gradient-descent';

% dw:
% 0: feature inversion
% 1: one class
exp.opts.task = 0;


exp.opts.init =[];
exp.opts.layer_ind =[];
exp.opts.dsp = 25;
exp.opts.do_dc = 0;
exp.opts.feats = [];
exp.opts.do_thres =0;
% for one class bp
exp.opts.mask = [];
exp.opts.momentum = 0.9;
exp.opts.grad_ran = [0 0];
% for inpainting
exp.opts.xmask = [];
exp.opts.freezeDropout = 1;
% for pool1 regu
exp.opts.regu =[];
% for feature mask
exp.opts.fmask =[];
% patch prior
exp.opts.pd=[];
% color whiten
exp.opts.im_w=[];
% nabla_2
exp.opts.lambdaTV2=0;
exp.opts.TV2beta=2;
% handy change data change
exp.opts.lambdaD=1;
% power spectrum prior
exp.opts.lambdaPS=0;
exp.opts.PS=[];
% dropout mask
exp.opts.m67=[];
% np patch init
exp.opts.np=[];

% f6 two net
exp.opts.net2=[];
exp.opts.mask_pre = {};

exp.opts.lambdaC = [];
exp.opts.regu_c = [];


[exp,varargin] = vl_argparse(exp, varargin) ;
exp.opts = vl_argparse(exp.opts, varargin) ;
