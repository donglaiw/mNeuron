function res=U_cnn_fb(net,res,x,opt,tmp_mask)
switch net.cnn_mode
    case 0
        % matcnn format
        switch opt
            case 0 % feedfoward
                res = vl_simplenn_dw(net, x) ;
            case 1 % feedfoward+backprop
                res = vl_simplenn_dw(net, x, single(1),res) ;
            case 2 % set dropout mask
                res = vl_simplenn_dw(net, x); % x is the random initialized image
                for j=1:numel(opts.regu)
                    res(opts.regu{j}{1}).aux(:)=0;
                    res(opts.regu{j}{1}).aux(:,:,find(opts.m67{j}))=2;
                end
        end
    case 1
        %if size(x,4)==1;x  = repmat(x,[1 1 1 10]);end
        scores = net.caffe.forward({x});
        scores = scores{1};
        % loss function
        switch net.task
            case 0 % feature inversion
                output_diff = -2*(net.feats-scores(:,:,:,1));
                res.y = scores;
                bb = net.caffe.backward({reshape(output_diff,size(scores))}); % gradient at the image space.
                res.x = sum(output_diff(:).^2);
            case 1% neuron visualization
                if exist('tmp_mask','var')&&~isempty(tmp_mask)
                    output_diff = - tmp_mask;
                else
                    output_diff = -net.layers{end}.mask;
                end
                bb = net.caffe.backward({reshape(output_diff,size(scores))}); % gradient at the image space.
                %res(end-1).x = squeeze(bb{1});
                res.y=[];
                res.x = scores(:)'*output_diff(:);
        end
        res.dzdx=bb{1};
end
