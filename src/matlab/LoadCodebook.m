function [ gene_to_seq, seq_to_gene ] = LoadCodebook( input_path, split_index, do_reverse )
% new_LoadCodebook

    % load file
    fname = fullfile(input_path, 'genes.csv');
    f = readmatrix(fname, 'OutputType', 'string', "Delimiter", ',');

    % load gene name and sequence 
    % f(:,1) - gene name, f(:,2) - gene barcode 
    if do_reverse
        f(:,2) = reverse(f(:,2));
    end

    for i=1:size(f, 1)
        f(i,2) = EncodeBases(f(i,2));
    end
    
    if ~isempty(split_index)
        f(:,2) = eraseBetween(f(:,2), split_index, split_index);

        % flip
        front = extractAfter(f(:,2), split_index-1);
        back = extractBefore(f(:,2), split_index);
        f(:,2) = front + back;

    end
    
    seq_to_gene = dictionary(f(:,2), f(:,1));
    gene_to_seq = dictionary(f(:,1), f(:,2));

