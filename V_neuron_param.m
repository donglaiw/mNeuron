function opts = V_neuron_param(task,mid,lid,opts)

switch task
    case 1
        % neuron inversion
        opts.do_thres=1; % threshold the output image [0 255] after each iteration
        opts.grad_ran=[0.1 100]; % adaptively scale the gradient scale if it's out of range
        switch mid
            case 1
                opts.lambdaL2 = 1e-10 ;opts.lambdaTV = 1e2;opts.lambdaD = 1e-3;
                opts.learningRate = 0.04 * [...
                    0.05 * ones(1,100), ...
                    0.01 * ones(1,100)];
                switch lid
                case 10
                    % p5, default
                case 13
                    % fc7
                opts.lambdaD = 3e-2;
                case 14
                    % fc8
                opts.lambdaD = 1e-2;
                end
            case 2
                opts.lambdaL2 = 1e-10 ;opts.lambdaTV = 8e2;opts.lambdaD = 3e-3;
                opts.learningRate = 0.001 * [...
                    0.05 * ones(1,100), ...
                    0.01 * ones(1,50)];
                opts.grad_ran=[0.1 100];
            case 3
                opts.lambdaL2 = 1e-10 ;opts.lambdaTV = 8e2;opts.lambdaD = 3e-3;
                opts.learningRate = 0.004 * [...
                    0.05 * ones(1,100), ...
                    0.01 * ones(1,50)];
            case 4
                % annealing schedule
                opts.lambdaL2 = [1e0*ones(1,50) 1e1*ones(1,50) 1e2*ones(1,100)];
                opts.lambdaTV = 1e-1*[5e2*ones(1,100) 5e2*ones(1,100)];
                opts.lambdaD = 3e-3;
                opts.learningRate = 0.04 * [...
                    0.05 * ones(1,100), ...
                    0.01 * ones(1,100)];
        end
    case 2
end
