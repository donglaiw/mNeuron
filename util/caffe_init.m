model_def_file ='';
switch mid
    case 1
        model_file = 'models/caffe_model/bvlc_reference_caffenet.caffemodel';
        % model_file = '/data/vision/torralba/deeplearning/CAMnet/alexnet/bvlc_reference_caffenet.caffemodel';
        ll= [2,4,8,9,10,11,12,13,14];
        nn = {'c1','n1','c3','c4','c5','p5','f6','f7','f8'};
        switch layer
            case num2cell(ll)
                model_def_file = ['models/caffe_model_def/deploy_alexnet_imagenet_1_' nn{ll==layer} '.prototxt'];
            case -1
                model_def_file = ['models/caffe_model_def/deploy_alexnet_imagenet_1.prototxt'];
        end
    case 2
        model_file='models/caffe_model/VGG_ILSVRC_16_layers.caffemodel';
        % model_file = '/data/vision/torralba/deeplearning/CAMnet/VGGnet/VGG_ILSVRC_16_layers.caffemodel';
        ll= [4,7,11,15,16,17,18,19,20,21,22];
        nn = {'p1','p2','p3','p4','c51','c52','c53','p5','f6','f7','f8'};
        switch layer
            case num2cell(ll)
                model_def_file = ['models/caffe_model_def/deploy_vgg16_imagenet_' nn{ll==layer} '.prototxt'];
            case -1
                model_def_file = ['models/caffe_model_def/deploy_vgg16_imagenet_1.prototxt'];
        end
    case 3
        model_file = ['models/caffe_model/nin_imagenet.caffemodel'];
        % model_file = '/data/vision/torralba/deeplearning/CAMnet/NIN/nin_imagenet.caffemodel';
        ll= [2,3,4,5,9,13,14,15,16,17];
        nn = {'c1','cp1','cp2','p0','p2','p3','c4','cp7','cp8','p4'};
        switch layer
            case num2cell(ll)
                model_def_file = ['models/caffe_model_def/deploy_nin_imagenet_' nn{ll==layer} '.prototxt'];
            case -1
                model_def_file = ['models/caffe_model_def/deploy_nin_imagenet_1.prototxt'];
        end
        
    case 4
        model_file = ['models/caffe_model/bvlc_googlenet.caffemodel'];
        % model_file = '/data/vision/torralba/deeplearning/CAMnet/googlenet_imagenet/bvlc_googlenet.caffemodel';
        ll= [45,9,13,14,15,16,17];
        nn = {'i4a','p2','p3','p3_d','p4'};
        switch layer
            case num2cell(ll)
                model_def_file = ['models/caffe_model_def/deploy_googlenet_imagenet_' nn{ll==layer} '.prototxt'];
            case -1
                model_def_file = ['models/caffe_model_def/deploy_googlenet_imagenet_1.prototxt'];
        end
end
if exist(model_file,'file') && exist(model_def_file,'file')
    %% loading the network
    caffe('reset');
    caffe('init', model_def_file, model_file,'test');
    if gpu_id>-1
        caffe.caffe.set_mode_gpu();caffe.set_device(gpu_id);
    else
        caffe.set_mode_cpu();
    end
    if layer==-1
        % get network information
        curImg = uint8(255*rand(256,256,3));
        if ~exist('IMAGE_MEAN','var')
            d = load('data/ilsvrc_2012_mean.mat');
            IMAGE_MEAN = d.image_mean;
        end
        scores = caffe('forward', {U_prepare_image(curImg, IMAGE_MEAN,mid,1)});
        response = caffe('get_all_layers');
        layernames = caffe('get_names');
    end
else
    error('caffe model file or model definition does not exist !')
end
