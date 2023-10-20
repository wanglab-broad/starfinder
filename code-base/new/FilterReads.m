function obj = FilterReads( obj, end_base )
% FilterReads

    % This is just as a sanity check; reads are actually filtered
    % by whether they are in the codebook
    color_seq = obj.signal.allSpots.color_seq;
    
    % older code for filtering reads in sequence space
    barcodes = DecodeCS(color_seq, end_base(1));
    % barocdes_with_correct_base_1 = startsWith(barcodes, end_base(1));
    % barocdes_with_correct_base_2 = endsWith(barcodes, end_base(2));
    % class(barocdes_with_correct_base_2)
    % barcodes_with_correct_form = barocdes_with_correct_base_1 && barocdes_with_correct_base_2;
    barcodes_with_correct_form = startsWith(barcodes, end_base(1)) & endsWith(barcodes, end_base(2));
    
    fprintf('Filtration Statistics:\n');
    score_1 = sum(barcodes_with_correct_form)/numel(barcodes);
    s = sprintf('%f [%d / %d] percent of good reads are %sNNNNN%s\n',...
        sum(barcodes_with_correct_form)/numel(barcodes),...
        sum(barcodes_with_correct_form),...
        numel(barcodes),...
        end_base(1),...
        end_base(2));
    fprintf(s);

    % filter reads based on codebook
    codebook_barcodes = obj.codebook.seqToGene.keys;
    barcodes_in_codebook = contains(color_seq, codebook_barcodes);
    
    score_2 = sum(barcodes_in_codebook)/numel(barcodes);
    s = sprintf('%f [%d / %d] percent of good reads are in codebook\n',...
        sum(barcodes_in_codebook)/numel(barcodes),...
        sum(barcodes_in_codebook),...
        numel(barcodes));
    fprintf(s);

    score_3 = sum(barcodes_in_codebook)/sum(barcodes_with_correct_form);
    s = sprintf('%f [%d / %d] percent of %sNNNNN%s reads are in codebook\n',...
        sum(barcodes_in_codebook)/sum(barcodes_with_correct_form),...
        sum(barcodes_in_codebook), ...
        sum(barcodes_with_correct_form),...
        end_base(1),...
        end_base(2));            
    fprintf(s);

    obj.signal.scores = [score_1 score_2 score_3]; 
    
    obj.signal.goodSpots = obj.signal.allSpots(barcodes_in_codebook, :);
    obj.signal.goodSpots{:, "barcode"} = barcodes(barcodes_in_codebook);
    obj.signal.goodSpots{:, "gene"} = obj.codebook.seqToGene(color_seq(barcodes_in_codebook));

end