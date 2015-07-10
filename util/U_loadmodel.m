% path to matcaffe
%addpath('../matlab/caffe');
addpath('/data/vision/torralba/gigaSUN/caffeCPU2/matlab/caffe');

% model
modelSetFolder = '/data/vision/torralba/deeplearning/CAMnet';

%mid=3;
switch mid
case 1
 netName = 'caffeNet_imagenet';
 model_file = [modelSetFolder '/alexnet/bvlc_reference_caffenet.caffemodel'];
 model_def_file = [modelSetFolder '/alexnet/deploy10.prototxt'];
case 2
 netName = 'VGG16_imagenet';
 model_file = [modelSetFolder '/VGGnet/VGG_ILSVRC_16_layers.caffemodel'];
 model_def_file = [modelSetFolder '/VGGnet/deploy_vgg16.prototxt'];
case 3
 netName = 'NIN';
 model_file = [modelSetFolder '/NIN/nin_imagenet.caffemodel'];
 model_def_file = [modelSetFolder '/NIN/deploy_nin_imagenet.prototxt'];
case 4
    netName = 'googlenetBVLC_imagenet';
model_file = [modelSetFolder '/googlenet_imagenet/bvlc_googlenet.caffemodel'];
model_def_file = [modelSetFolder '/googlenet_imagenet/deploy.protxt'];
end

%% loading the network
caffe('reset');
caffe('init', model_def_file, model_file,'test');
%caffe('set_mode_cpu');
caffe('set_device',1);

%load imagenet_val
d = load('/data/vision/torralba/small-projects/bolei_deep/caffe/ilsvrc_2012_mean.mat');
IMAGE_MEAN = d.image_mean;
curImg = uint8(255*rand(256,256,3));
%curImg = imread([DIM names{1}{1}]);
%[height_original, weight_original, ~] = size(curImg);
%keyboard
scores = caffe('forward', {U_prepare_image(curImg, IMAGE_MEAN,mid)});
response = caffe('get_all_layers');
layernames = caffe('get_names');
szs = cell2mat(arrayfun(@(x) size(response{x})',1:numel(response),'UniformOutput',false));
return


weights = caffe('get_weights');
weights_LR = squeeze(weights(end,1).weights{1,1});
bias_LR = weights(end,1).weights{2,1};
netInfo = cell(size(layernames,1),3);
for i=1:size(layernames,1)
    netInfo{i,1} = layernames{i};
    netInfo{i,2} = i;
    netInfo{i,3} = size(response{i,1});
end

weightInfo = cell(size(weights,1),1);
for i=1:size(weights,1)
    weightInfo{i,1} = weights(i,1).layer_names;
    weightInfo{i,2} = weights(i,1).weights{1,1};
    weightInfo{i,3} = size(weights(i,1).weights{1,1});
end

%load('place_nImageNet.mat');
d = load('/data/vision/torralba/small-projects/bolei_deep/caffe/ilsvrc_2012_mean.mat');
IMAGE_MEAN = d.image_mean;
IMAGE_DIM = 256;
CROPPED_DIM = netInfo{1,3}(1); 

%% predict the image

%curImg = imread('7.jpg');
load imagenet_val
load im_cnn
curImg = imread([DIM names{1}{1}]);
[height_original, weight_original, ~] = size(curImg);
%keyboard
scores = caffe('forward', {U_prepare_image(curImg, IMAGE_MEAN,mid)});
response = caffe('get_all_layers');
scoresMean = mean(squeeze(scores{1}),2);
[value_category, IDX_category] = sort(scoresMean,'descend');


disp('top 5 predictions');
for i=1:5
    disp([place_n{IDX_category(i)} ' ' num2str(value_category(i))]);
end

%% backpropagation to compute the gradient img
scores = squeeze(scores{1,1});
for j=1:5
    clear output_diff
    output_diff = zeros(1,1,size(scores,1),size(scores,2),'single');
    output_diff(1,1,IDX_category(j),1:10) = 1;
    %output_diff(1,1,:,:) = scores;
    gradients = caffe('backward', {output_diff}); % gradient at the image space.
    allgradients = caffe('get_all_layerdiffs'); % gradients at all layer
    gradients = gradients{1};
    
    [alignImgMean alignImgSet] = U_crop2img(gradients);
    alignImgMean = double(alignImgMean);
    alignImgMean = imresize(alignImgMean, [height_original weight_original]);
    alignImgMean = imfilter(alignImgMean,fspecial('ave', [4 4]));
    alignImgMean = alignImgMean./max(alignImgMean(:));
    imshow(alignImgMean),colormap(jet);
    %heatMapSet(:,:,j) = alignImgMean;
    subplot(1,2,1),imshow(curImg),title(place_n{IDX_category(j)}),
    subplot(1,2,2),imshow(alignImgMean),colormap(jet);
    waitforbuttonpress
end
