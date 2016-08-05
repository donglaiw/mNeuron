function res = invert_nn_dw(net, ref, varargin)

% setup param
[net,opts,x] = invert_nn_pre(net, ref, varargin);
x0_size = cat(2,net.normalization.imageSize,opts.numRepeats);
x_momentum = zeros(x0_size, 'single') ;
load('x0_sigma.mat', 'x0_sigma');
y0 = repmat(ref, [1 1 1 opts.numRepeats]);
switch opts.task
    case 0;y0_sigma = norm(squeeze(y0(find(net.layers{end}.mask(:))))) ;
    case {1,2};y0_sigma = 1;
end


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
% visualize dropout mask
if ~isempty(opts.m67)
    res = U_cnn_fb(net,res,x,2);
end

E=zeros(5,opts.maxNumIterations);
for t=1:opts.maxNumIterations

  % 1. Effectively does both forward and backward passes
  res = U_cnn_fb(net,res,x,1,opts.mask);

 % The current best feature we could generate
  switch net.cnn_mode
  case 'matconvnet'; y = res(end-1).x ; 
  case 'caffe'; y = res.y ; 
  end
  dr = zeros(size(x),'single'); % The derivative

  % 2. update the mask
  if ~isempty(opts.regu)
    for j=1:numel(opts.regu)
        regu_out{j} = res(opts.regu{j}{1}).x;
    end
  end
 
  % 3. calcuate gradient
  % 3.1 Data term
  E(1,t) = opts.lambdaD*res(end).x/(y0_sigma^2);

  % 3.2 Cost and derivative for TV\beta norm
  if opts.lambdaTV > 0 
    tmp_x =x;
      if ~isempty(opts.im_w)
        tmp_x=reshape(reshape(x,[],3)*opts.im_w',size(x));
      end
    [r_,dr_] = tv(tmp_x,opts.TVbeta,opts.xmask) ;
    E(2,t) = opts.lambdaTV/2 * r_ ;
    dr = dr + opts.lambdaTV/2 * dr_ ;
  end
  % 3.2 Cost and derivative of L\beta norm
  if opts.lambdaL2 > 0 
     tmp_x=x;
      if ~isempty(opts.im_w)
        tmp_x=reshape(reshape(x,[],3)*opts.im_w',size(x));
      end
    if ~isempty(opts.xmask)
        tmp_x = bsxfun(@times,x,opts.xmask);
    end
    if ~isempty(opts.regu_c)
      tmp_x = reshape(bsxfun(@minus,reshape(tmp_x,[],3),opts.regu_c),size(x));
    end    
    r_ = sum(tmp_x(:).^opts.beta) ;
    dr_ = opts.beta * tmp_x.^(opts.beta-1) ;

    E(3,t) = opts.lambdaL2/2 * r_ ;
    dr = dr + opts.lambdaL2/2 * dr_ ;
  end

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

    % gradient clipping
    tmp_max =max(abs(x_momentum(:)));
    if opts.grad_ran(1)>0
        if tmp_max>opts.grad_ran(2)
            x_momentum=x_momentum/tmp_max*opts.grad_ran(2);
        elseif tmp_max<opts.grad_ran(1)
            x_momentum=x_momentum/tmp_max*opts.grad_ran(1);
        end
    end

    if opts.dsp>=0
      fprintf('iter:%05d: max_grad=%3.4g, err_data=%5.4g, err_TV=%5.4g, err_L2=%5.4g, err_all=%5.4g;\n', t, tmp_max,E(1,t),E(2,t),E(3,t), E(4,t)) ;
  end

  % This is the main update step (we are updating the the variable
  % along the gradient
    if ~isempty(opts.xmask)
        x_momentum = bsxfun(@times,x_momentum,opts.xmask);
    end
    x = x + x_momentum ;

  % truncate output 
  if opts.do_thres
      tmp_m = opts.denormalize(x);
      tmp_m(tmp_m<0)=0;
      tmp_m(tmp_m>255)=255;
      x = opts.normalize(tmp_m);
  end

  %% -----------------------------------------------------------------------
  %% Plots - Generate several plots to keep track of our progress
  %% -----------------------------------------------------------------------

  if opts.dsp>0 && mod(t-1,opts.dsp)==0
    figure(1) ; clf ;
    output{end+1} = opts.denormalize(x) ;
    subplot(3,2,[1 3]) ;
    if opts.numRepeats > 1
      vl_imarraysc(output{end}) ;
    else
      imagesc(uint8(output{end})) ;
    end
    axis image ; 

    switch opts.task
    case 0 % feature inversion
        subplot(3,2,2) ;
        len = min(1000, numel(y0));
        a = squeeze(y0(1:len)) ;
        b = squeeze(y(1:len)) ;
        plot(1:len,a,'b'); hold on ;
        plot(len+1:2*len,abs(b-a), 'r');
        legend('\Phi_0', '|\Phi-\Phi_0|') ;
        title(sprintf('reconstructed layer %s', ...
          net.layerName)) ;
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
    end

    subplot(3,2,6) ;
    if ~isempty(opts.xmask)
        hist(reshape(x(opts.xmask==1),[],1),100);
    else
        hist(x(:),100) ;
    end
    grid on ;
    title('histogram of x') ;
   
    subplot(3,2,5) ;
    plot(E') ;
    h = legend('recon', 'tv_reg', 'l2_reg', 'tot') ;
    set(h,'color','none') ; grid on ;
    title(sprintf('iter:%d \\lambda_{tv}:%g \\lambda_{l2}:%g rate:%g obj:%s', ...
                  t, opts.lambdaTV, opts.lambdaL2, lr, opts.objective)) ;

   drawnow ;

  end % end if(mod(t-1,25) == 0)
end % end loop over maxNumIterations
if opts.dsp<=0
    output{end+1} = opts.denormalize(x) ;
end

% Compute the features optained using feedforward on the computed inverse
res_nn = vl_simplenn_dw(net, x);

clear res;
res.input = NaN;
res.output = output ;
res.energy = E ;
res.y0 = y0 ;
res.y = res_nn(end-1).x ;
res.opts = opts ;
res.err = res_nn(end).x / y0_sigma^2 ;

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
