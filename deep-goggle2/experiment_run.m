function res= experiment_run(exp,net)

addpath('/data/vision/torralba/gigaSUN/caffeCPU2/matlab/caffe');
modelSetFolder = '/data/vision/torralba/deeplearning/CAMnet';
if ~exist('net','var')
    switch exp.model
        case 'caffe-ref'
            net = load('networks/imagenet-caffe-ref.mat') ;
            exp.opts.normalize = get_cnn_normalize(net.normalization) ;
            exp.opts.denormalize = get_cnn_denormalize(net.normalization) ;
        case 'caffe-mitplaces'
            net = load('places-caffe-ref-upgraded.mat');
            exp.opts.normalize = get_cnn_normalize(net.normalization) ;
            exp.opts.denormalize = get_cnn_denormalize(net.normalization) ;
        case 1
            im_sz=227;
        case 2
            im_sz=224;
        case 3
            im_sz=224;
        case 4
            im_sz=224;
    end
else
    if ~isfield(exp.opts,'normalize') || isempty(exp.opts.normalize)
        exp.opts.normalize = get_cnn_normalize(net.normalization) ;
        exp.opts.denormalize = get_cnn_denormalize(net.normalization) ;
    end
end

net.cnn_id = 0;

if isinf(exp.layer), exp.layer = numel(net.layers) ; end
if ismember(exp.model,1:4)
    tmp_l = cell(1,exp.layer);
    switch exp.model
    case {1,3,4}
        d = load('data/ilsvrc_2012_mean.mat');
        mm = d.image_mean;
    case 2
        mm = repmat(reshape([103.939, 116.779, 123.68],[1,1,3]),[256 256]);
    end 
    net=struct('normalization',struct('imageSize',[im_sz im_sz 3]),'im_mean',mm);
    net.layers = tmp_l;
    net.cnn_id = exp.model;
    exp.opts.normalize = @(x,im_mean) U_prepare_image(x, im_mean,exp.model,-2);
    exp.opts.denormalize = @(x,im_mean) U_prepare_image(x, im_mean,exp.model,-1);
end

net.layers = net.layers(1:exp.layer) ;
if ~isempty(exp.opts.net2)
    exp.opts.net2.layers=exp.opts.net2.layers(1:exp.layer);
end
args = expandOpts(exp.opts) ;
net.task = exp.opts.task;
switch exp.opts.task
    case {1,2}
        % one or two class
        if ~isempty(exp.opts.layer_ind)
            net.layers=net.layers(exp.opts.layer_ind);
            if ~isempty(exp.opts.net2)
                exp.opts.net2.layers=exp.opts.net2.layers(exp.opts.layer_ind);
            end
        end
        
        if ismember(exp.model,1:4)
            % invert caffe model
            net.layer = exp.layer;
            res = invert_nn_caffe(net, exp.opts.mask, args{:}) ;
        else
            % invert matcaffe model
            res = invert_nn_dw(net, exp.opts.mask, args{:}) ;
        end
    case 0
        % same as deep-goggle
        % feature inversion
        if isempty(exp.opts.feats)
            im = imread(exp.path) ;
            if size(im,3) == 1, im = cat(3,im,im,im) ; end
            exp.opts.feats = compute_features(net, im);
        end
        if(strcmp(exp.opts.optim_method , 'gradient-descent'))
            if isempty(exp.opts.layer_ind) ||  exp.opts.layer_ind(1)==1
                % d layer/ d I
                if ismember(exp.model,1:6)
                    net.feats = exp.opts.feats;
                    net.layer = exp.layer;
                    res = invert_nn_caffe(net, exp.opts.feats, args{:}) ;
                else
                    res = invert_nn(net, exp.opts.feats, args{:}) ;
                end
            else
                % d layer_1/ d layer_2
                net.layers=net.layers(exp.opts.layer_ind);
                res = invert_nn_layer(net, exp.opts.feats, args{:}) ;
            end
        else
            fprintf(1, 'Unknown optimization method %s\n', exp.opts.optim_method);
            return;
        end
end


end
function args = expandOpts(opts)
% -------------------------------------------------------------------------
args = horzcat(fieldnames(opts), struct2cell(opts))' ;
end


