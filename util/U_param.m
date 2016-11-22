function opts = U_param(opts,mn,ln)

opts.do_thres=1; % threshold the output image [0 255] after each iteration
opts.grad_ran=[0.1 100]; % adaptively scale the gradient scale if it's out of range
switch opts.task
    case 0 % feature inversion
        switch mn
            case 'alex'
            opts.lambdaL2 = 8e-10 ;opts.lambdaTV = 1e0;opts.lambdaD = 1e2;
            opts.learningRate = 0.0004 * [...
                ones(1,100), ...
                0.1 * ones(1,200), ...
                0.01 * ones(1,100)];
              switch ln
                case 'p5'; % default
                opts.lambdaL2 = 8e-8 ;opts.lambdaTV = 1e2;opts.lambdaD = 1e1;
            opts.learningRate = 0.0004 * [...
                ones(1,500), ...
                0.1 * ones(1,200), ...
                0.01 * ones(1,100)];
                end
            end
    case 1 % neuron inversion
        switch mn
            case 'alex'
                opts.lambdaL2 = 1e-10 ;opts.lambdaTV = 1e2;opts.lambdaD = 1e-3;
                opts.learningRate = 0.04 * [...
                    0.05 * ones(1,100), ...
                    0.01 * ones(1,100)];
                switch ln
                case 'p5'; % default
                case 'f7'; opts.lambdaD = 3e-2;
                case 'f8'; opts.lambdaD = 1e-2;
                end
            case 'vgg16'
                opts.lambdaL2 = 1e-10 ;opts.lambdaTV = 8e2;opts.lambdaD = 3e-3;
                opts.learningRate = 0.001 * [...
                    0.05 * ones(1,100), ...
                    0.01 * ones(1,50)];
                opts.grad_ran=[0.1 100];
            case 'nin'
                opts.lambdaL2 = 1e-10 ;opts.lambdaTV = 8e2;opts.lambdaD = 3e-3;
                opts.learningRate = 0.004 * [...
                    0.05 * ones(1,100), ...
                    0.01 * ones(1,50)];
            case 'gnet'
                % annealing schedule
                opts.lambdaL2 = [1e0*ones(1,50) 1e1*ones(1,50) 1e2*ones(1,100)];
                opts.lambdaTV = 1e-1*[5e2*ones(1,100) 5e2*ones(1,100)];
                opts.lambdaD = 3e-3;
                opts.learningRate = 0.04 * [...
                    0.05 * ones(1,100), ...
                    0.01 * ones(1,100)];
        end
    case 2
        % image inpainting
        switch mn
        case 'alex'
            opts.lambdaL2 = 1e-8 ;opts.lambdaTV = 5e2; 
            opts.lambdaD = 5e1;
            opts.grad_ran=[0 20];
            opts.learningRate = 0.004 * [...
            0.05 * ones(1,150), ...
            0.05 * ones(1,0), ...
            0.05 * ones(1,0)];
        end
end
