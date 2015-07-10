function res=U_cnn_fb(net,res,x,opt,tmp_mask)
switch net.cnn_id
    case 0
        % matcnn format
        switch opt
            case 1
                % feedfoward+backprop
                res = vl_simplenn_dw(net, x, single(1),res) ;
            case 2
                % set dropout mask
                res = vl_simplenn_dw(net, x); % x is the random initialized image
                res(19).aux(:)=0;
                res(19).aux(:,:,find(opts.m67{1}))=2;
                res(22).aux(:)=0;
                res(22).aux(:,:,find(opts.m67{2}))=2;
        end
        
    case {1,2,3,4}
        %if size(x,4)==1;x  = repmat(x,[1 1 1 10]);end
        scores = caffe('forward', {x});
        ff = caffe('get_all_layers');
        % loss function
        switch net.task
            case 0
                % inversion
                output_diff = -2*(net.feats-ff{net.layer}(:,:,:,1));
                res.y = ff{net.layer}(:,:,:,1);
                %bb = caffe('backward', {repmat(output_diff,[1,1,1,10])}); % gradient at the image space.
                bb = caffe('backward', {reshape(output_diff,size(ff{net.layer}))}); % gradient at the image space.
                res.x = sum(output_diff(:).^2);
            case 1
                if exist('tmp_mask','var')&&~isempty(tmp_mask)
                    output_diff = - tmp_mask;
                else
                    output_diff = -net.layers{end}.mask;
                end
                bb = caffe('backward', {reshape(output_diff,size(ff{net.layer}))}); % gradient at the image space.
                res.x = sum(reshape(ff{net.layer}(:,:,:,1),[],1).*output_diff(:));
                res.y = squeeze(ff{net.layer}(:,:,:,1));
        end
        res.dzdx=bb{1}(:,:,:,1);
end
