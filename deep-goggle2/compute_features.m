function feats = compute_features(net, img, opt,do_norm)
% feats = compute_features(net, img) - compute features from a network
%
%  Inputs: net - the neural network (same as for vl_simplenn)
%          img - the input image (H x W x 3 - RGB image)
%
%  Output: feats - the reference given as argument to invert_nn.m
%
% Author: Aravindh Mahendran
%      New College, University of Oxford

% normalize the input image

if ~exist('do_norm','var');do_norm=1;end
if do_norm
    normalize = get_cnn_normalize(net.normalization);
    x0 = normalize(img);
else
    x0 = single(img);
end
% Convert the image into a 4D matrix as required by vl_simplenn
x0 = repmat(x0, [1, 1, 1, 1]);
% Run feedforward for network

if exist('opt','var') && opt(1)~=-1
    net.layers=net.layers(1:max(opt)-1);
end
res = vl_simplenn(net, x0);
if ~exist('opt','var')
    feats = res(end).x;
else
    if opt(1)==-1
        opt = 1:numel(res);
    end
    feats = cell(1,numel(opt));
    for i=1:numel(opt)
        feats{i} = res(opt(i)).x;
    end
   
    % cell to array
    if numel(opt)==1
        feats= feats{1};
    end
end
end
