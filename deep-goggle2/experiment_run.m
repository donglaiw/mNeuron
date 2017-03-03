function res= experiment_run(exp,net)

net.cnn_id = exp.model;
switch net.cnn_mode
case 0 % matconvnet
    exp.opts.normalize = get_cnn_normalize(net.normalization) ;
    exp.opts.denormalize = get_cnn_denormalize(net.normalization) ;
case 1
    switch exp.model
    case {'alex','nin','gnet'} % alexnet, nin, inception
        d = load('data/ilsvrc_2012_mean.mat');
        mm = d.image_mean;
    case 'vgg16' % vgg
        mm = repmat(reshape([103.939, 116.779, 123.68],[1,1,3]),[256 256]);
    case 'alexWeb'
        mm = repmat(reshape([0.8309, 0.8310, 0.8328],[1,1,3]),[256 256]);
    end 
    net.normalization=struct('imageSize',[net.im_sz net.im_sz 3],'averageImage',mm);
    exp.opts.normalize = @(x) U_prepare_image(x, mm,exp.model,-2);
    exp.opts.denormalize = @(x) U_prepare_image(x, mm,exp.model,-1);
end

net.task = exp.opts.task;
switch exp.opts.task
case {1,2}
    % 1: neuron inversion
    % 2: neuron inpainting
    res = invert_nn_dw(net, exp.opts.mask, exp.opts) ;
case 0
    % feature inversion
    % same as deep-goggle
    if ischar(exp.opts.feats)
        im = single(imread(exp.opts.feats));
        if size(im,3) == 1, im = cat(3,im,im,im) ; end
        exp.opts.feats = compute_features(net, exp.opts.normalize(im),0);
        %keyboard
    end
    % gradient-descent
    net.feats = exp.opts.feats;
    res = invert_nn_dw(net, exp.opts.feats, exp.opts) ;
end


end
function args = expandOpts(opts)
% -------------------------------------------------------------------------
args = horzcat(fieldnames(opts), struct2cell(opts))' ;
end


