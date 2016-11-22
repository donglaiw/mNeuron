
mNames={'alex','vgg16','nin','gnet'};

mid = find(cellfun(@(x) strcmp(x,mn),mNames));
if isempty(mid)
    error('unexpected model name "mn". should be one of alex,vgg16,nin,gnet ');
end
switch cnn_mode 
    case 'matconvnet'
        % matconvnet
        addpath([MATCONVNET_ROOT '/matlab/mex'])
        addpath(genpath(MATCONVNET_ROOT))
        switch mn
        case 'alex'
            net = load('models/imagenet-caffe-ref.mat') ;
            % convert ln into lid
            ll= [1,4,9,11,13,15,16,18,20];
            nn = {'c1','n1','c3','c4','c5','p5','f6','f7','f8'};
            lid = ll(cellfun(@(x) strcmp(x,ln),nn));
            if opts.task==1
                net.layers(ll(ll>lid))=[];
            end
            net.layer=lid;
            net.im_sz=227;
            test_img = randn([net.im_sz net.im_sz 3],'single');
            res = vl_simplenn_dw(net, test_img);
            net.layer_sz=size(res(end).x);
        end
        net.cnn_mode=0;
    case 'caffe'
        % matcaffe
        net.cnn_mode=1;
        net.im_sz=224;
        addpath([CAFFE_ROOT '/matlab']);
        % load truncated caffe model at layer lid
        model_def_file = '';
        model_file = '';
        switch mn
            case 'alex'
                net.im_sz=227;
                model_file = 'models/bvlc_alexnet.caffemodel';
                model_def_file = ['models/caffe_model_def/deploy_alexnet_imagenet_' ln '.prototxt'];
            case 'vgg16'
                model_file='models/VGG_ILSVRC_16_layers.caffemodel';
                model_def_file = ['models/caffe_model_def/deploy_vgg16_imagenet_' ln '.prototxt'];
            case 'nin'
                model_file = ['models/nin_imagenet.caffemodel'];
                model_def_file = ['models/caffe_model_def/deploy_nin_imagenet_' ln '.prototxt'];
            case 'gnet'
                model_file = ['models/bvlc_googlenet.caffemodel'];
                model_def_file = ['models/caffe_model_def/deploy_googlenet_imagenet_' ln '.prototxt'];
        end
        if exist(model_file,'file') && exist(model_def_file,'file')
            %% loading the network
            caffe.reset_all();
            net.caffe = caffe.Net( model_def_file, model_file,'test');
            if gpu_id>-1
                caffe.set_mode_gpu();caffe.set_device(gpu_id);
            else
                caffe.set_mode_cpu();
            end
            test_img = randn([net.im_sz net.im_sz 3],'single');
            res = net.caffe.forward({test_img});
            net.layer_sz=size(res{1});
        else
            if ~exist(model_file,'file') 
                error('%s does not exist !',model_file)
            elseif ~exist(model_def_file,'file')
                error('%s does not exist !',model_def_file)
            end
        end
end 
net.layerName=ln;


