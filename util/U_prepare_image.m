function images = U_prepare_image(im,IMAGE_MEAN,cnn_name,opt)
% ------------------------------------------------------------------------
if ~exist('IMAGE_MEAN','var') || isempty(IMAGE_MEAN)
    switch cnn_name
    case {'alex','nin','gnet'} % alexnet, nin, inception
        d = load('data/ilsvrc_2012_mean.mat');
        IMAGE_MEAN = d.image_mean;
    case 'vgg16'
        IMAGE_MEAN = repmat(reshape([103.939, 116.779, 123.68],[1,1,3]),[256 256]);
    end
end
if ~exist('opt','var');opt=0;end

IMAGE_DIM = 256;
switch opt
    case -1
        % visual: denormalize
        switch cnn_name
            case 'alex';CROPPED_DIM = 227;
            case {'vgg16','nin','gnet'};CROPPED_DIM = 224;
        end
        indices = [0 IMAGE_DIM-CROPPED_DIM] + 1;
        center = floor(indices(2) / 2)+1;
        images = bsxfun(@plus,permute(im(:,:,[3 2 1],:),[2,1,3,4]) , IMAGE_MEAN(center:center+CROPPED_DIM-1,center:center+CROPPED_DIM-1,3:-1:1));
    case -2
        % visual: normalize
        switch cnn_name
            case 'alex';CROPPED_DIM = 227;
            case {'vgg16','nin','gnet'};CROPPED_DIM = 224;
        end
        indices = [0 IMAGE_DIM-CROPPED_DIM] + 1;
        center = floor(indices(2) / 2)+1;
        images = bsxfun(@minus,im(:,:,[3 2 1],:) , IMAGE_MEAN(center:center+CROPPED_DIM-1,center:center+CROPPED_DIM-1,:));
        images = permute(images,[2,1,3,4]);
    case {0,1,5,6}
        switch cnn_name
            case 'alex'
                % alexnet
                CROPPED_DIM = 227;
                % resize to fixed input size
                im = single(im);
                im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
                % permute from RGB to BGR (IMAGE_MEAN is already BGR)
                im = im(:,:,[3 2 1]) - IMAGE_MEAN;
                
                % oversample (4 corners, center, and their x-axis flips)
                images = zeros(CROPPED_DIM, CROPPED_DIM, 3, 10, 'single');
                indices = [0 IMAGE_DIM-CROPPED_DIM] + 1;
                curr = 1;
                for i = indices
                    for j = indices
                        images(:, :, :, curr) = ...
                            permute(im(i:i+CROPPED_DIM-1, j:j+CROPPED_DIM-1, :), [2 1 3]);
                        images(:, :, :, curr+5) = images(end:-1:1, :, :, curr);
                        curr = curr + 1;
                    end
                end
                center = floor(indices(2) / 2)+1;
                images(:,:,:,5) = ...
                    permute(im(center:center+CROPPED_DIM-1,center:center+CROPPED_DIM-1,:), ...
                    [2 1 3]);
                images(:,:,:,10) = images(end:-1:1, :, :, curr);
            case 'vgg16'
                CROPPED_DIM = 224;
                
                % resize to fixed input size
                im = single(im);
                
                if size(im, 1) < size(im, 2)
                    im = imresize(im, [IMAGE_DIM NaN]);
                else
                    im = imresize(im, [NaN IMAGE_DIM]);
                end
                
                % RGB -> BGR
                im = im(:, :, [3 2 1]);
                
                % oversample (4 corners, center, and their x-axis flips)
                images = zeros(CROPPED_DIM, CROPPED_DIM, 3, 10, 'single');
                
                indices_y = [0 size(im,1)-CROPPED_DIM] + 1;
                indices_x = [0 size(im,2)-CROPPED_DIM] + 1;
                center_y = floor(indices_y(2) / 2)+1;
                center_x = floor(indices_x(2) / 2)+1;
                
                
                % hack the mean image
                IMAGE_MEAN =  repmat(reshape([103.939, 116.779, 123.68],[1,1,3]),[CROPPED_DIM CROPPED_DIM]);
                curr = 1;
                for i = indices_y
                    for j = indices_x
                        images(:, :, :, curr) = ...
                            permute(im(i:i+CROPPED_DIM-1, j:j+CROPPED_DIM-1, :)-IMAGE_MEAN, [2 1 3]);
                        images(:, :, :, curr+5) = images(end:-1:1, :, :, curr);
                        curr = curr + 1;
                    end
                end
                images(:,:,:,5) = ...
                    permute(im(center_y:center_y+CROPPED_DIM-1,center_x:center_x+CROPPED_DIM-1,:)-IMAGE_MEAN, ...
                    [2 1 3]);
                images(:,:,:,10) = images(end:-1:1, :, :, curr);
        end
        
        if opt==1
            images = images(:,:,:,5);
        end
        
end
