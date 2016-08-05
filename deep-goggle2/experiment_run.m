function res= experiment_run(exp,net)

net.cnn_id = exp.model;
switch net.cnn_mode
case 0 % matconvnet
    exp.opts.normalize = get_cnn_normalize(net.normalization) ;
    exp.opts.denormalize = get_cnn_denormalize(net.normalization) ;
case 1
    switch exp.model
    case {1,3,4}
        d = load('data/ilsvrc_2012_mean.mat');
        mm = d.image_mean;
    case 2
        mm = repmat(reshape([103.939, 116.779, 123.68],[1,1,3]),[256 256]);
    end 
    net.normalization=struct('imageSize',[net.im_sz net.im_sz 3],'averageImage',mm);
    exp.opts.normalize = @(x) U_prepare_image(x, mm,exp.model,-2);
    exp.opts.denormalize = @(x) U_prepare_image(x, mm,exp.model,-1);
end

args = expandOpts(exp.opts) ;
net.task = exp.opts.task;
switch exp.opts.task
    case {1,2}
        res = invert_nn_dw(net, exp.opts.mask, args{:}) ;
    case 0
        % feature inversion
        % same as deep-goggle
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


