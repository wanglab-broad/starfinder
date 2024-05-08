function obj = FilterReadsMultiSegment( obj, end_base, split_index )
% FilterReads

    % This is just as a sanity check; reads are actually filtered
    % by whether they are in the codebook
    end_base_char = [];
    for i=1:numel(end_base)
        end_base_char = [end_base_char; char(end_base{i})];
    end
    color_seq = obj.signal.allSpots.color_seq;

    color_seq_char = char(color_seq);
    segments = {};
    for n=1:numel(split_index)
        current_index = split_index(n);
        if n==1
            segments{1} = color_seq_char(:, 1:current_index-1, 1);
            segments{end+1} = color_seq_char(:, current_index:end, 1);
        elseif n==numel(split_index)
            segments{end+1} = color_seq_char(:, previous_index:current_index-1, 1);
            segments{end+1} = color_seq_char(:, current_index:end, 1);
        else
            segments{end+1} = color_seq_char(:, previous_index:current_index-1, 1);
        end
        previous_index = current_index+1;
    end
    % segments_length = cellfun("length", segments)
    % color_seq_char_segments = mat2cell(color_seq_char, 1, segments_length, 2)
    
    barcodes = strings(numel(color_seq), numel(segments));
    for i=1:numel(segments)
        current_segment_char = segments{i};
        current_segment_string = string(current_segment_char);
        current_barcodes = DecodeCS(current_segment_string, end_base_char(i, 1));
        barcodes(:, i) = current_barcodes;
    end

    fprintf('Filtration Statistics:\n');

    for i=1:numel(end_base)

        current_end_base_char = end_base_char(i, :);
        barcodes_with_correct_form = startsWith(barcodes(:, i), current_end_base_char(1)) & endsWith(barcodes(:, i), current_end_base_char(2));
    
        current_score = sum(barcodes_with_correct_form)/numel(color_seq);
        s = sprintf('%f [%d / %d] percent of good reads are %s...%s\n',...
            sum(barcodes_with_correct_form)/numel(color_seq),...
            sum(barcodes_with_correct_form),...
            numel(color_seq),...
            current_end_base_char(1),...
            current_end_base_char(2));
        fprintf(s);
    end

    % filter reads based on codebook
    codebook_barcodes = obj.codebook.seqToGene.keys;
    barcodes_in_codebook = contains(color_seq, codebook_barcodes);
    
    score_1 = sum(barcodes_in_codebook)/numel(color_seq);
    s = sprintf('%f [%d / %d] percent of good reads are in codebook\n',...
        sum(barcodes_in_codebook)/numel(color_seq),...
        sum(barcodes_in_codebook),...
        numel(color_seq));
    fprintf(s);

    obj.signal.scores = [score_1]; 
    
    obj.signal.goodSpots = obj.signal.allSpots(barcodes_in_codebook, :);
    % obj.signal.goodSpots{:, "barcode"} = barcodes(barcodes_in_codebook);
    obj.signal.goodSpots{:, "gene"} = obj.codebook.seqToGene(color_seq(barcodes_in_codebook));

end