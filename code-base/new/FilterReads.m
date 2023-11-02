function obj = FilterReads( obj, end_base )
% FilterReads

    % This is just as a sanity check; reads are actually filtered
    % by whether they are in the codebook
    end_base_char = [];
    for i=1:numel(end_base)
        end_base_char = [end_base_char; char(end_base{i})];
    end
    color_seq = obj.signal.allSpots.color_seq;

    barcodes = DecodeCS(color_seq, end_base_char(1, 1));

    fprintf('Filtration Statistics:\n');
    % filter reads based on codebook
    codebook_barcodes = obj.codebook.seqToGene.keys;
    barcodes_in_codebook = contains(color_seq, codebook_barcodes);
    
    score_1 = sum(barcodes_in_codebook)/numel(barcodes);
    s = sprintf('%f [%d / %d] percent of good reads are in codebook\n',...
        sum(barcodes_in_codebook)/numel(barcodes),...
        sum(barcodes_in_codebook),...
        numel(barcodes));
    fprintf(s);

    obj.signal.scores = [score_1]; 

    for i=1:numel(end_base)

        current_end_base_char = end_base_char(i, :);
        barcodes_with_correct_form = startsWith(barcodes, current_end_base_char(1)) & endsWith(barcodes, current_end_base_char(2));
    
        current_score_2 = sum(barcodes_with_correct_form)/numel(barcodes);
        s = sprintf('%f [%d / %d] percent of good reads are %sNNNNN%s\n',...
            sum(barcodes_with_correct_form)/numel(barcodes),...
            sum(barcodes_with_correct_form),...
            numel(barcodes),...
            current_end_base_char(1),...
            current_end_base_char(2));
        fprintf(s);

        current_score_3 = sum(barcodes_in_codebook)/sum(barcodes_with_correct_form);
        s = sprintf('%f [%d / %d] percent of %sNNNNN%s reads are in codebook\n',...
            sum(barcodes_in_codebook)/sum(barcodes_with_correct_form),...
            sum(barcodes_in_codebook), ...
            sum(barcodes_with_correct_form),...
            current_end_base_char(1),...
            current_end_base_char(2));            
        fprintf(s);

        obj.signal.scores = [obj.signal.scores current_score_2 current_score_3]; 
    end
    
    obj.signal.goodSpots = obj.signal.allSpots(barcodes_in_codebook, :);
    obj.signal.goodSpots{:, "barcode"} = barcodes(barcodes_in_codebook);
    obj.signal.goodSpots{:, "gene"} = obj.codebook.seqToGene(color_seq(barcodes_in_codebook));

end