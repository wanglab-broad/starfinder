function barcodes = DecodeCS( color_seq, start_base )
%DecodeCS

    % get dims
    Npoint = numel(color_seq);
    
    % preallocating
    barcodes = string();
    
    % construct reverse hash table for decoding
    color_to_base = dictionary();  
    color_to_base{'1'} = {'AA','CC','GG','TT'};
    color_to_base{'2'} = {'AC','CA','GT','TG'};
    color_to_base{'3'} = {'AG','CT','GA','TC'};
    color_to_base{'4'} = {'AT','CG','GC','TA'};

    for i=1:Npoint

        current_color_seq = char(color_seq(i));
        current_barcode = '';
        ref_base = start_base;

        for j=1:numel(current_color_seq)
            possible_segments = color_to_base{current_color_seq(j)};
            for n=1:4

                p = possible_segments{n};
                if strcmp(p(1), ref_base) && isempty(current_barcode)
                    current_barcode = p;
                elseif strcmp(p(1), ref_base)
                    current_barcode = [current_barcode p(2)];
                end

            end 

            ref_base = current_barcode(end);
            
        end
        
        barcodes(i) = string(current_barcode);
        
    end
