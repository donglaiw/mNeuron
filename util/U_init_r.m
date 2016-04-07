function im_init=U_init_r(im_init,mask,std_rgb,off_rgb)
if ~exist('std_rgb','var');std_rgb=10;end
if ~exist('off_rgb','var');off_rgb=[0 0 0];end

if size(std_rgb,2)==1
    im_init(mask==1) = std_rgb*randn(1,nnz(mask==1));
else
    if size(std_rgb,3)==1
        std_rgb=repmat(std_rgb,[1,1,3]);
    end
    im_init(mask==1) = std_rgb(mask==1).*randn(nnz(mask==1),1);
end


tmp_im = reshape(im_init,[],3);

if size(off_rgb,3)==1
tmp_im(mask(:,:,1)==1,1) = tmp_im(mask(:,:,1)==1,1)+off_rgb(1); 
tmp_im(mask(:,:,1)==1,2) = tmp_im(mask(:,:,1)==1,2)+off_rgb(2); 
tmp_im(mask(:,:,1)==1,3) = tmp_im(mask(:,:,1)==1,3)+off_rgb(3); 
else
    tmp_off = reshape(off_rgb,[],3);
tmp_im(mask(:,:,1)==1,1) = tmp_im(mask(:,:,1)==1,1)+tmp_off(mask(:,:,1)==1,1); 
tmp_im(mask(:,:,1)==1,2) = tmp_im(mask(:,:,1)==1,2)+tmp_off(mask(:,:,1)==1,2); 
tmp_im(mask(:,:,1)==1,3) = tmp_im(mask(:,:,1)==1,3)+tmp_off(mask(:,:,1)==1,3); 
end
im_init = reshape(tmp_im,size(im_init));


