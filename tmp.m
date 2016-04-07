switch tid
case 1
    for doid=1:6
        load(['data/inpaint/locate_' num2str(doid)],'im0','mask_rm'); 
        mm = uint8(repmat(imdilate(mask_rm,strel('disk',0)),[1,1,3]));
        mm([1 end],:,:)=0;mm(:,[1 end],:)=0;
        imwrite(uint8(im0.*(1-mm)),['data/inpaint/locate_pm_' num2str(doid) '.png'])
    end

end
