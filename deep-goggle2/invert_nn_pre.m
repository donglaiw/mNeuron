function [net,opts,x] = invert_nn_dw_pre(net, ref, tmp_var)
% INVERT  Invert a CNN representation

opts.learningRate = 0.001*[...
    ones(1,800), ...
    0.1 * ones(1,500), ...
    0.01 * ones(1,500), ...
    0.001 * ones(1,200), ...
    0.0001 * ones(1,100) ] ;
[opts, tmp_var] = vl_argparse(opts, tmp_var) ;

opts.maxNumIterations = numel(opts.learningRate) ;
opts.objective = 'l2' ; % The experiments in the paper use only l2
opts.lambdaTV = 10 ; % Coefficient of the TV^\beta regularizer
opts.lambdaL2 = 0.1 ; % Coefficient of the L\beta regularizer on the reconstruction
opts.TVbeta = 1; % The power to which TV norm is raized.
opts.beta = 4 ; % The \beta of the L\beta regularizer
opts.momentum = 0.9 ; % Momentum used in the optimization
opts.numRepeats = 1 ; % Number of reconstructions to generate
opts.normalize = [] ; % A function handle that normalizes network input
opts.denormalize = [] ; % A function handle that denormalizes network input
opts.dropout = 0.5; % Dropout rate for any drop out layers.
% The L1 loss puts a drop out layer. We didn't experiment with this though.
opts.filterGroup = NaN ; % Helps select one or the other group of filters.
% This is used in selecting filters in conv 1 of alex net to see the difference
% in their properties.
opts.neigh = +inf ; % To select a small neighborhood of neurons.
opts.optim_method = 'gradient-descent'; % Only 'gradient-descent' is currently used
% I really want to use L-BFGS or CG later.

% dw:
opts.task = 0;
% 1. for feature inversion
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
% for feature mask
opts.fmask =[];
% patch prior
opts.pd=[];
% color normalize
opts.im_w=[];
% nabla_2
opts.lambdaTV2=0;
opts.TV2beta=2;
% handy change data change
opts.lambdaD=1;
% power spectrum prior
opts.lambdaPS=0;
opts.PS=[];
% dropout mask
opts.m67=[];
% np patch init
opts.np=[];

% f6 two net
opts.net2=[];
% initialize with full opening
opts.mask_pre = {};

% color regu
opts.lambdaC = [];
opts.regu_c = [];

% Parse the input arguments to override the above defaults
opts = vl_argparse(opts, tmp_var) ;

% Update the number of iterations using the learning rate if required
if isinf(opts.maxNumIterations)
    opts.maxNumIterations = numel(opts.learningRate) ;
end

net.regu = opts.regu;
net.fmask = opts.fmask;
% The size of the image that we are trying to obtain
x0_size = cat(2,net.normalization.imageSize,opts.numRepeats);
%keyboard

% x0_sigma is computed using a separate dataset.
% This is a useful normalization that helps scale the different terms in the
% optimization.
load('x0_sigma.mat', 'x0_sigma');

% Replicate the feature into a block. This is used for multiple inversions in parallel.
y0 = repmat(ref, [1 1 1 opts.numRepeats]);

% initial inversion image of size x0_size
if  ~isempty(opts.init)
    x= opts.init;
else
    x = randn(x0_size, 'single') ;
    %x= imfilter(x,fspecial('gaussian',50,30));
    %x = ones(x0_size, 'single') ;
    %x = zeros(x0_size, 'single') ;
    x = x / norm(x(:)) * x0_sigma  ;
end
x_momentum = zeros(x0_size, 'single') ;

switch opts.task
    case {0,-1}
        % allow reconstructing a subset of the representation by setting
        % a suitable mask on the features y0
        sf = 1:size(y0,3) ;
        if opts.filterGroup == 1
            sf= vl_colsubset(sf, 0.5, 'beginning') ;
        elseif opts.filterGroup == 2 ;
            sf= vl_colsubset(sf, 0.5, 'ending') ;
        end
        nx = min(opts.neigh, size(y0,2)) ;
        ny = min(opts.neigh, size(y0,1)) ;
        sx = (0:nx-1) + ceil((size(y0,2)-nx+1)/2) ;
        sy = (0:ny-1) + ceil((size(y0,1)-ny+1)/2) ;
        mask = zeros(size(y0), 'single') ;
        mask(sy,sx,sf,:) = 1 ;
        y0_sigma = norm(squeeze(y0(find(mask(:))))) ;
    case {1,-2,2}
        %mask = opts.mask;
        mask = ref;
        y0_sigma = 1;
end

%% Tweak the network by adding a reconstruction loss at the end

layer_num = numel(net.layers) ; % The layer number which we are reconstructing
% This is saved here just for printing our progress as optimization proceeds

switch opts.objective
    case 'l2'
        % Add the l2 loss over the network
        ly.type = 'custom' ;
        ly.w = y0 ;
        ly.mask = mask ;
        ly.forward = @nndistance_forward ;
        ly.backward = @nndistance_backward ;
        net.layers{end+1} = ly ;
    case 'l1'
        % The L1 loss might want to use a dropout layer.
        % This is just a guess and hasn't been tried.
        ly.type = 'dropout' ;
        ly.rate = opts.dropout ;
        net.layers{end+1} = ly ;
        ly.type = 'custom' ;
        ly.w = y0 ;
        ly.mask = mask ;
        ly.forward = @nndistance1_forward ;
        ly.backward = @nndistance1_backward ;
        net.layers{end+1} = ly ;
    case 'inner'
        % The inner product loss may be suitable for some networks
        ly.type = 'custom' ;
        ly.w = - y0 .* mask ;
        ly.forward = @nninner_forward ;
        ly.backward = @nninner_backward ;
        net.layers{end+1} = ly ;
    case 'oneclass'
        % The inner product loss may be suitable for some networks
        % maxize the probability of one class
        ly.type = 'custom' ;
        ly.mask =  mask ;
        ly.forward = @oneclass_forward ;
        ly.backward = @oneclass_backward ;
        net.layers{end+1} = ly ;
    otherwise
        error('unknown opts.objective') ;
end

% --------------------------------------------------------------------
function res_ = oneclass_forward(ly, res, res_)
% --------------------------------------------------------------------
% the value for that one class
res_.x = -sum(reshape(res.x.*oneclass(ly.mask,res.x),1,[]));

% --------------------------------------------------------------------
function res = oneclass_backward(ly, res, res_)
% --------------------------------------------------------------------
% positive or negative ?
res.dzdx = -oneclass(ly.mask,res.x);

function out=oneclass(mask,res)
out = reshape(mask,size(res));
%out = repmat(reshape(mask,[1,1,numel(mask)]),[size(res,1),size(res,2),1]);

% --------------------------------------------------------------------
function res_ = nndistance_forward(ly, res, res_)
% --------------------------------------------------------------------
res_.x = nndistance(res.x, ly.w, ly.mask) ;

% --------------------------------------------------------------------
function res = nndistance_backward(ly, res, res_)
% --------------------------------------------------------------------
res.dzdx = nndistance(res.x, ly.w, ly.mask, res_.dzdx) ;

% --------------------------------------------------------------------
function y = nndistance(x,w,mask,dzdy)
% --------------------------------------------------------------------
if nargin <= 3
    d = x - w ;
    y = sum(sum(sum(sum(d.*d.*mask)))) ;
else
    y = dzdy * 2 * (x - w) .* mask ;
end

% --------------------------------------------------------------------
function res_ = l1loss_forward(ly, res, res_)
% --------------------------------------------------------------------
res_.x = ly.w*sum(abs(res.x(:)));


% --------------------------------------------------------------------
function res = l1loss_backward(ly, res, res_)
% --------------------------------------------------------------------
res.dzdx = zeros(size(res.x), 'single');
res.dzdx(res.x > 0) = single(ly.w)*res_.dzdx;
res.dzdx(res.x < 0) = -single(ly.w)*res_.dzdx;
res.dzdx(res.x == 0) = single(0);

% --------------------------------------------------------------------
function res_ = nndistance1_forward(ly, res, res_)
% --------------------------------------------------------------------
res_.x = nndistance1(res.x, ly.w, ly.mask) ;

% --------------------------------------------------------------------
function res = nndistance1_backward(ly, res, res_)
% --------------------------------------------------------------------
res.dzdx = nndistance1(res.x, ly.w, ly.mask, res_.dzdx) ;

% --------------------------------------------------------------------
function y = nndistance1(x,w,mask,dzdy)
% --------------------------------------------------------------------
if nargin <= 3
    d = x - w ;
    y = sum(sum(sum(sum(abs(d).*mask)))) ;
else
    y = dzdy * sign(x - w) .* mask ;
end

% --------------------------------------------------------------------
function res_ = nninner_forward(ly, res, res_)
% --------------------------------------------------------------------
res_.x = nninner(res.x, ly.w) ;

% --------------------------------------------------------------------
function res = nninner_backward(ly, res, res_)
% --------------------------------------------------------------------
res.dzdx = nninner(res.x, ly.w, res_.dzdx) ;

% --------------------------------------------------------------------
function y = nninner(x,w,dzdy)
% --------------------------------------------------------------------
if nargin <= 2
    y = sum(sum(sum(sum(w.*x)))) ;
else
    y = dzdy * w ;
end


