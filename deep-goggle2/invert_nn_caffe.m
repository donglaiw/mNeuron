function res = invert_nn_dw_caffe(net, ref, varargin)

% setup param
[net,opts,x] = invert_nn_pre(net, ref, varargin);

x0_size = cat(2,net.normalization.imageSize,opts.numRepeats);
x_momentum = zeros(x0_size, 'single') ;
load('x0_sigma.mat', 'x0_sigma');
y0 = repmat(ref, [1 1 1 opts.numRepeats]);
switch opts.task
    case 0
        y0_sigma = norm(y0(:)) ;
    case {1,2}
        y0_sigma = 1;
end
layer_num = numel(net.layers) ; % The layer number which we are reconstructing


%% --------------------------------------------------------------------
%%                                                 Perform optimisation
%% --------------------------------------------------------------------

% Run forward propogation once on the modified network before we
% begin backprop iterations - this is to exploit an optimization in
% vl_simplenn


% recored results
output = {} ;
prevlr = 0 ;

% Iterate until maxNumIterations to optimize the objective
% and generate the reconstuction
regu_out=[];
if ~isempty(opts.regu)
    regu_out=cell(1,numel(opts.regu));
end
res=[];

if ~isempty(opts.m67)
    res = U_cnn_fb(net,res,x,2);
end

mask_switch=[0,0];
mask_tmp =[];
if ~isempty(opts.mask_pre)
    mask_switch = opts.mask_pre{1};
    mask_tmp =opts.mask_pre{2};
end

for t=1:opts.maxNumIterations
    % Effectively does both forward and backward passes
    if t==mask_switch(1)
        mask_tmp = opts.mask;
    end
    if t==mask_switch(2)
        % remove bg
        [aa,bb] = bwlabel(sum(abs(dr_d),3)<1e-6);
        [cc,dd]=histc(aa(:),1:bb);
        [~,ee] = max(cc);
        mm = imerode(aa==ee,strel('disk',10));
        xx=reshape(x,[],3);
        xx(mm,:)=0;
        x=reshape(xx,size(x));
    end
    res = U_cnn_fb(net,res,x,1,mask_tmp);
    
    y = res.y ; % The current best feature we could generate
    
    dr = zeros(size(x),'single'); % The derivative
    
    tmp_TV = opts.lambdaTV(1);
    if numel(opts.lambdaTV)>=t
        tmp_TV = opts.lambdaTV(t);
    end
    if tmp_TV>0% Cost and derivative for TV\beta norm
        tmp_x =x;
        if ~isempty(opts.im_w)
            tmp_x=reshape(reshape(x,[],3)*opts.im_w',size(x));
        end
        [r_,dr_] = tv(tmp_x,opts.TVbeta,opts.xmask) ;
        E(2,t) = tmp_TV/2 * r_ ;
        dr = dr + tmp_TV/2 * dr_ ;
    else
        E(2,t) = 0;
    end
    
    tmp_L2 = opts.lambdaL2(1);
    if numel(opts.lambdaL2)>=t
        tmp_L2 = opts.lambdaL2(t);
    end
    if tmp_L2 > 0 % Cost and derivative of L\beta norm
        tmp_x=x;
        if ~isempty(opts.im_w)
            tmp_x=reshape(reshape(x,[],3)*opts.im_w',size(x));
        end
        if ~isempty(opts.xmask)
            tmp_x = bsxfun(@times,x,opts.xmask);
        end
        r_ = sum(tmp_x(:).^opts.beta) ;
        dr_ = opts.beta * tmp_x.^(opts.beta-1) ;
        E(3,t) = tmp_L2/2 * r_ ;
        dr = dr + tmp_L2/2 * dr_ ;
    else
        E(3,t) = 0;
    end
    
    
    % Rescale the different costs and add them up
    %if t>25;keyboard;end
    E(1,t) = opts.lambdaD*res(end).x/(y0_sigma^2);
    
    
    E(2:3,t) = E(2:3,t) / (x0_sigma^2) ;
    E(4,t) = sum(E(1:3,t)) ;
    lr = opts.learningRate(min(t, numel(opts.learningRate))) ;
    
    % when the learning rate changes suddently, it is not
    % possible for the gradient to crrect the momentum properly
    % causing the algorithm to overshoot for several iterations
    if lr ~= prevlr
        fprintf('switching learning rate (%f to %f) and resetting momentum\n', ...
            prevlr, lr) ;
        x_momentum = 0 * x_momentum ;
        prevlr = lr ;
    end
    
    % x_momentum combines the current gradient and the previous gradients
    % with decay (opts.momentum)
    dr_d = opts.lambdaD*res(1).dzdx;
    %keyboard
    % [max(dr_d(:)) max(dr(:)/(x0_sigma^2)) max(opts.lambdaD*res(1).dzdx(:))]
    switch opts.task
        case 0
            dsp_p=x0_sigma^2/y0_sigma^2;
            x_momentum = opts.momentum * x_momentum ...
                - lr * dr ...
                - (lr * x0_sigma^2/y0_sigma^2) * dr_d;
        case {1,2}
            dsp_p=x0_sigma^2;
            x_momentum = opts.momentum * x_momentum ...
                - lr * dr ...
                - (lr *  (x0_sigma^2)*dr_d);
    end
    
    % adapt gradient
    % on pix val
    tmp_max =max(abs(x_momentum(:)));
    if opts.grad_ran(1)>0
        if tmp_max>opts.grad_ran(2)
            x_momentum=x_momentum/tmp_max*opts.grad_ran(2);
        elseif tmp_max<opts.grad_ran(1)
            x_momentum=x_momentum/tmp_max*opts.grad_ran(1);
        end
    end
    
    if opts.dsp>=0
        fprintf('iter:%05d sq. max_grad: %3.4g, err:%5.4g, %5.4g, %5.4g; obj:%5.4g;\n', t, tmp_max,E(1,end),E(2,end),E(3,end), E(4,end)) ;
    end
    
    
    
    
    % This is the main update step (we are updating the the variable
    % along the gradient
    if ~isempty(opts.xmask)
        x_momentum = bsxfun(@times,x_momentum,opts.xmask);
    end
    switch opts.do_dc
        case 0
            % direct grad
            x = x + x_momentum ;
        case 1
            % low pass
            %disp([t mean(x_momentum(:))])
            % almost zero out
            %for i=1:3;x_momentum(:,:,i)=conv2(x_momentum(:,:,i),fspecial('gaussian',[10 10],5),'same') ;end
            % bad smoothness
            %for i=1:3;x_momentum(:,:,i)=medfilt2(x_momentum(:,:,i),[3 3]) ;end
            x = x + x_momentum;
        case 2
            % pyramid
            d_loss_pyr = Zpyr;
            for i=1:3;d_loss_pyr(:,i) = buildGpyr_matrix(double(x_momentum(:,:,i)), pyramid_Zmeta);end
            Zpyr = Zpyr +d_loss_pyr;
            for i=1:3;x(:,:,i) = reconLpyr_matrix(Zpyr(:,i), pyramid_Zmeta);end
    end
    
    %if t==80;imwrite(uint8(x+net.normalization.averageImage),'db_pm.png');return;end
    
    if opts.do_thres
        tmp_m = opts.denormalize(x,net.im_mean);
        tmp_m(tmp_m<0)=0;
        tmp_m(tmp_m>255)=255;
        x = opts.normalize(tmp_m,net.im_mean);
        % max(x(:)-x0(:))
        % x0=x;tmp_m = opts.denormalize(x,net.im_mean);x = opts.normalize(tmp_m,net.im_mean); max(x(:)-x0(:))
    end
    
    %% -----------------------------------------------------------------------
    %% Plots - Generate several plots to keep track of our progress
    %% -----------------------------------------------------------------------
    
    if opts.dsp>0 && mod(t-1,opts.dsp)==0
        figure(1) ; clf ;
        
        output{end+1} = opts.denormalize(x,net.im_mean) ;
        subplot(3,2,[1 3]) ;
        if opts.numRepeats > 1
            vl_imarraysc(output{end}) ;
        else
            %imagesc(vl_imsc(output{end})) ;
            imagesc(uint8(output{end})) ;
        end
        axis image ;
        
        switch opts.task
            case 0
                subplot(3,2,2) ;
                len = min(1000, numel(y0));
                a = squeeze(y0(1:len)) ;
                b = squeeze(y(1:len)) ;
                plot(1:len,a,'b'); hold on ;
                plot(len+1:2*len,abs(b-a), 'r');
                legend('\Phi_0', '|\Phi-\Phi_0|') ;
                title(sprintf('reconstructed layer %d %s', ...
                    layer_num, ...
                    net.layers{layer_num}.type)) ;
                legend('ref', 'delta') ;
                
                if ~isempty(opts.regu)
                    subplot(3,2,6) ;
                    a= mean(reshape(regu_out{1},[],size(regu_out{1},3)));
                    b= opts.regu{1}{2};
                    len = min(1000, numel(a));
                    plot(1:len,a,'b'); hold on ;
                    plot(len+1:2*len,abs(b-a), 'r');
                    axis tight
                    title(opts.regu{1}{1})
                end
            case 1
                if isfield(opts.np,'lambdaP') && opts.np.lambdaP(t)>0
                    subplot(3,2,[2 4]) ;
                    imagesc(uint8(opts.denormalize(dsp_im,net.im_mean)));
                    axis image ;
                end
        end
        
        subplot(3,2,6) ;
        if ~isempty(opts.xmask)
            hist(reshape(x(opts.xmask==1),[],1),100);
        else
            hist(x(:),100) ;
        end
        grid on ;
        title('histogram of x') ;
        %{
    subplot(3,2,5) ;
    plot(E') ;
    if t>1
        h = legend('recon', 'tv_reg', 'l2_reg', 'tot') ;
        set(h,'color','none') ; grid on ;
    end
    title(sprintf('iter:%d \\lambda_{tv}:%g \\lambda_{l2}:%g rate:%g obj:%s', ...
                  t, tmp_TV, tmp_L2, lr, opts.objective)) ;
        %}
        drawnow ;
        
    end % end if(mod(t-1,25) == 0)
end % end loop over maxNumIterations
if opts.dsp<=0
    output{end+1} = opts.denormalize(x,net.im_mean) ;
end

% Compute the features optained using feedforward on the computed inverse
%res_nn = vl_simplenn_dw(net, x);
res_nn = res;

clear res;
res.input = NaN;
res.output = output ;
res.energy = E ;
res.y0 = y0 ;
res.y = res_nn.x ;
res.opts = opts ;
res.err = res_nn.x / y0_sigma^2 ;

% --------------------------------------------------------------------
function [e, dx] = tv(x,beta,mask)
% dw: increase only boundary weight doesn't help
% which equivalently push the boundary inside by 1-pix
% --------------------------------------------------------------------
if(~exist('beta', 'var'))
    beta = 1; % the power to which the TV norm is raized
end
if(~exist('mask', 'var'));mask=[];end

d1 = x(:,[2:end end],:,:) - x ;
d2 = x([2:end end],:,:,:) - x ;
if ~isempty(mask)
    new_m = imdilate(mask,strel('disk',1))>0;
    d1 = bsxfun(@times,d1, new_m);
    d2 = bsxfun(@times,d2, new_m);
end
v = sqrt(d1.*d1 + d2.*d2).^beta ;
e = sum(sum(sum(sum(v)))) ;
if nargout > 1
    d1_ = (max(v, 1e-5).^(2*(beta/2-1)/beta)) .* d1;
    d2_ = (max(v, 1e-5).^(2*(beta/2-1)/beta)) .* d2;
    % take grad from both end
    %  x_{i,j}-x_{i,j+1}
    %  x_{i,j-1}-x_{i,j}
    d11 = d1_(:,[1 1:end-1],:,:) - d1_ ;
    d22 = d2_([1 1:end-1],:,:,:) - d2_ ;
    d11(:,1,:,:) = - d1_(:,1,:,:) ;
    d22(1,:,:,:) = - d2_(1,:,:,:) ;
    dx = beta*(d11 + d22);
    if ~isempty(mask)
        dx = bsxfun(@times,dx,mask);
    end
    if(any(isnan(dx)))
    end
end

% --------------------------------------------------------------------
function test_tv()
% --------------------------------------------------------------------
x = randn(5,6,1,1) ;
[e,dr] = tv(x,6) ;
vl_testder(@(x) tv(x,6), x, 1, dr, 1e-3) ;

function [e, dx, pp0_r] = patch_np(x,pd,mask,opt,t)
switch pd.opt_d
    case {2,3}
        % patchmatch
        % need to change into 0-1 float
        algo = 'cputiled';cores=8;niter=5;
        if pd.opt_d==3
            %tmp_db = bsxfun(@rdivide,bsxfun(@minus,pd.db,pd.db_m),pd.db_std);
            % pre-processed for speed
            tmp_db = pd.db;
            x_m = reshape(mean(reshape(x,[],3)),[1,1,3]);
            x_std = reshape(std(reshape(x,[],3)),[1,1,3]);
            tmp_x = bsxfun(@rdivide,bsxfun(@minus,x,x_m),x_std);
            tmp_ran = [min(min(tmp_db(:)),min(tmp_x(:))),max(max(tmp_db(:)),max(tmp_x(:)))];
            tmp_x=(tmp_x-tmp_ran(1))/range(tmp_ran);
            tmp_db=(tmp_db-tmp_ran(1))/range(tmp_ran);
        else
            tmp_ran = [min(min(pd.db(:)),min(x(:))),max(max(pd.db(:)),max(x(:)))];
            tmp_x=(x-tmp_ran(1))/range(tmp_ran);
            tmp_db=(pd.db-tmp_ran(1))/range(tmp_ran);
        end
        %[tmp_dx,tmp_dy] = gradient(tmp_x);
        %[tmp_db_dx,tmp_db_dy] = gradient(tmp_db);
        
        %ann = nnmex(cat(3,tmp_x,tmp_dx,tmp_dy)/2+1, cat(3,tmp_db,tmp_db_dx,tmp_db_dy)/2+1, algo, pd.psz, niter, [], [], [], [], cores);
        ann = nnmex(tmp_x, tmp_db, algo, pd.psz, niter, [], [], [], [], cores);
        if isfield(pd,'wavg') && pd.wavg>0
            ann2 = ann(1:end-pd.psz+1,1:end-pd.psz+1,:);
            bw = exp(-pd.wavg*double(ann2(:,:,3))/(2*0.1*255^2*9));%
            % can't use weight ...
            num_p = size(tmp_db,1)-pd.psz+1;
            b_id = ann2(:,:,1)*num_p+ann2(:,:,2)+1;
            b_pp =U_getpatch_im(tmp_db,b_id,pd.psz,1);
            pp0_r = U_col2im(b_pp,pd.psz,size(tmp_x),1,'waverage',bw);
        else
            pp0_r = single(votemex(tmp_db, ann,[],algo,pd.psz))/255;
        end
        pp0_r = tmp_ran(1)+pp0_r*range(tmp_ran);
        
        if pd.opt_d==3
            pp0_r = bsxfun(@plus,bsxfun(@times,pp0_r,pd.db_std),pd.db_m);
        end
        %if t==100; keyboard;end
        %pp0_r2 = U_col2im(b_pp,pd.psz,size(tmp_x),1,'average');
        %figure(2),subplot(121),U_im(255*x),subplot(122),U_im(255*pp0_r)
    case {0,1}
        pp0 = U_im2col(x,pd.psz,pd.psd,1);
        mask_ind =1:size(pp0,2);
        if ~isempty(mask)
            mask_ind = find(sum(U_im2col(mask,pd.psz,pd.psd,1))>1);
            pp0 = pp0(:,mask_ind);
        end
        switch pd.opt_d
            case 0
                dist= pdist2(pd.db',pp0');
            case 1
                pp0_m = mean(pp0,1);
                pp0_s = std(pp0,[],1)+eps;
                pp = bsxfun(@rdivide,bsxfun(@minus,pp0,pp0_m),pp0_s);
                dist= pdist2(pd.db',pp');
        end
        
        [dist_val,dist_id] = min(dist,[],1);
        switch pd.opt_d
            case 0
                new_p = pd.db(:,dist_id);
            case 1
                new_p = bsxfun(@plus,bsxfun(@times,pd.db(:,dist_id),pp0_s),pp0_m);
        end
        bw=ones(size(dist_val));
        if isfield(pd,'wavg') && pd.wavg>0
            % can't use weight ...
            bw = 1./(dist_val.^pd.wavg);
        end
        
        switch pd.opt_v
            case 0
                % no-mask
                pp0_r = U_col2im(new_p,pd.psz,size(x),pd.psd,'average');
            case 1
                % no-mask
                pp0_r = U_col2im(b_pp,pd.psz,size(x),pd.psd,'waverage',bw);
            case 2
                sz_db=size(pd.db);
                pp0_r = T_syn_pm({reshape(new_p,[sz_db(1:3) numel(mask_ind)]),pd.opt_algo,pd.opt_avg});
            case 3
                im_sz=size(x);
                im_sz2=prod(im_sz(1:2));
                mat_ind = reshape(1:prod(im_sz(1:2)),im_sz(1:2));
                mat_ind = mat_ind(1:pd.psd:end-pd.psz+1,1:pd.psd:end-pd.psz+1);
                pind = reshape(bsxfun(@plus,(0:pd.psz-1)',(0:pd.psz-1)*im_sz(1)),1,[]);
                pp0_r=zeros([im_sz2 3],'single');
                count=zeros(im_sz2,1,'uint8');
                for i=1:numel(mask_ind)
                    mid= mask_ind(i);
                    pp0_r(mat_ind(mid)+pind,:) = pp0_r(mat_ind(mid)+pind,:) + reshape(new_p(:,i),[],3);
                    count(mat_ind(mid)+pind) = count(mat_ind(mid)+pind) + 1;
                end
                pp0_r(count>0,:) = pp0_r(count>0,:)./single(repmat(count(count>0),[1,3]));
                pp0_r=reshape(pp0_r,size(x));
                %if t==50;keyboard;end
                %{
        pp0_r2= reshape(pp0_r,[],3);
        tmp_x = reshape(x,[],3);
        %pp0_r2(count==0,:) = tmp_x(count==0,:);
        pp0_r2=reshape(pp0_r2,size(x));
        figure(2),U_im(255*[x reshape(pp0_r2,size(x))])
        figure(2),U_im(114+[x reshape(pp0_r2,size(x))])
        keyboard
        % weird patch
        figure(3),montage(uint8(255*reshape(new_p,7,7,3,[])))
                %}
        end
end

%{
        out = U_col2im(pp0-new_p,pd.psz,size(x),pd.psd,'average');
        U_im(255*[x out])
%}
dx = x-pp0_r;
if isfield(pd,'im_w_inv') && ~isempty(pd.im_w_inv)
    dx = reshape(reshape(dx,[],3)*pd.im_w_inv',size(dx));
    pp0_r = reshape(reshape(pp0_r,[],3)*pd.im_w_inv',size(dx));
end
dx(isnan(dx))=0;

if ~isempty(mask)
    dx = dx.*mask;
end
e = sum((dx(:)/2).^2);

