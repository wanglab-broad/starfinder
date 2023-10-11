function output_img = RegisterImagesGlobal( input_img, ref_idx, nblocks, useOverlay, log_file )
% 
    output_reg = zeros(size(obj.rawImages), 'uint8');
                output_reg(:,:,:,:,p.Results.ref_round) = obj.rawImages(:,:,:,:,p.Results.ref_round);
                
    rounds = 1:obj.Nround;
    rounds = rounds(rounds ~= p.Results.ref_round);
    
    if p.Results.useGPU
        
        for r=rounds
            tic;
%                     ref_round = gpuArray(obj.rawImages(:,:,:,:,p.Results.ref_round));
%                     fix = max(ref_round, [], 4);

            ref_round = obj.rawImages(:,:,:,:,p.Results.ref_round);
            fix = gpuArray(max(ref_round, [], 4));
            
%                     curr_round = gpuArray(obj.rawImages(:,:,:,:,r));
%                     curr_mov = max(curr_round, [], 4);
            
            curr_round = obj.rawImages(:,:,:,:,r);
            curr_mov = gpuArray(max(curr_round, [], 4));

            params = DFTRegister3D(fix, curr_mov, false);
            % disp("DFTRegister success!");
            for c=1:4
                curr_reg = DFTApply3D(gpuArray(curr_round(:,:,:,c)), params, false);
                curr_round(:,:,:,c) = curr_reg;
            end

            output_reg(:,:,:,:,r) = gather(curr_round);
            fprintf(sprintf('Round %d vs. Round %d finished [time=%02f]\n', r, p.Results.ref_round, toc));
            fprintf(obj.log, sprintf('Round %d vs. Round %d finished [time=%02f]\n', r, p.Results.ref_round, toc));
            fprintf(sprintf('Shifted by %s\n', num2str(params.shifts)));
            fprintf(obj.log, sprintf('Shifted by %s\n', num2str(params.shifts)));
            reset(gpuDevice);
        end
    
    else
        for r=rounds
            starting = tic;
            ref_round = obj.rawImages(:,:,:,:,p.Results.ref_round);
            fix = max(ref_round, [], 4);

            curr_round = obj.rawImages(:,:,:,:,r);
            curr_mov = max(curr_round, [], 4);

            params = DFTRegister3D(fix, curr_mov, false);
            % disp("DFTRegister success!");
            fprintf(sprintf('DFT register finished [time=%02f]\n', toc(starting)));
            
            starting_apply = tic;
            for c=1:4
                curr_reg = DFTApply3D(curr_round(:,:,:,c), params, false);
                curr_round(:,:,:,c) = curr_reg;
            end
            fprintf(sprintf('DFT apply finished [time=%02f]\n', toc(starting_apply)));
            
            output_reg(:,:,:,:,r) = curr_round;
            fprintf(sprintf('Round %d vs. Round %d finished [time=%02f]\n', r, p.Results.ref_round, toc(starting)));
            fprintf(obj.log, sprintf('Round %d vs. Round %d finished [time=%02f]\n', r, p.Results.ref_round, toc(starting)));
            fprintf(sprintf('Shifted by %s\n', num2str(params.shifts)));
            fprintf(obj.log, sprintf('Shifted by %s\n', num2str(params.shifts)));
        end
        
    end

end