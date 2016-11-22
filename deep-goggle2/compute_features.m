function feats = compute_features(net, img, do_norm)
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

switch net.cnn_mode
    case 0;res = vl_simplenn_dw(net, x0);feats = res(end).x;
    case 1;res = net.caffe.forward({x0});feats = res{end};
end
