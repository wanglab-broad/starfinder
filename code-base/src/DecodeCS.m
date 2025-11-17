function barcodes = DecodeCS( color_seq, start_base )
%DecodeCS

    % get dims
    Npoint = numel(color_seq);
    
    % preallocating
    barcodes = string();
    
    % construct reverse hash table for decoding
    % color_to_base = dictionary();  
    % color_to_base{'1'} = {'AA','CC','GG','TT'};
    % color_to_base{'2'} = {'AC','CA','GT','TG'};
    % color_to_base{'3'} = {'AG','CT','GA','TC'};
    % color_to_base{'4'} = {'AT','CG','GC','TA'};

    base_to_color = dictionary();  
    base_to_color{'AA'} = '1';
    base_to_color{'CC'} = '1';
    base_to_color{'GG'} = '1';
    base_to_color{'TT'} = '1';

    base_to_color{'AC'} = '2';
    base_to_color{'CA'} = '2';
    base_to_color{'GT'} = '2';
    base_to_color{'TG'} = '2';

    base_to_color{'AG'} = '3';
    base_to_color{'CT'} = '3';
    base_to_color{'GA'} = '3';
    base_to_color{'TC'} = '3';

    base_to_color{'AT'} = '4';
    base_to_color{'CG'} = '4';
    base_to_color{'GC'} = '4';
    base_to_color{'TA'} = '4';

    % color_to_base = dictionary();  
    % color_to_base{'1'} = {'AT','TA','GC','CG'};
    % color_to_base{'2'} = {'AC','CA','GT','TG'};
    % color_to_base{'3'} = {'AA','TT','CC','GG'};
    % color_to_base{'4'} = {'AG','GA','CT','TC'};

    % base_to_color = dictionary();  
    % base_to_color{'AT'} = '1';
    % base_to_color{'TA'} = '1';
    % base_to_color{'GC'} = '1';
    % base_to_color{'CG'} = '1';

    % base_to_color{'AC'} = '2';
    % base_to_color{'CA'} = '2';
    % base_to_color{'GT'} = '2';
    % base_to_color{'TG'} = '2';

    % base_to_color{'AA'} = '3';
    % base_to_color{'TT'} = '3';
    % base_to_color{'GG'} = '3';
    % base_to_color{'CC'} = '3';

    % base_to_color{'AG'} = '4';
    % base_to_color{'GA'} = '4';
    % base_to_color{'CT'} = '4';
    % base_to_color{'TC'} = '4';

    keys_list = keys(dictionary(base_to_color));
    vals = values(dictionary(base_to_color));

    for i=1:Npoint

        current_color_seq = char(color_seq(i));
        current_barcode = '';
        ref_base = start_base;

        for j=1:numel(current_color_seq)
            % possible_segments = color_to_base{current_color_seq(j)};
            possible_segments = keys_list(cellfun(@(x) isequal(x, current_color_seq(j)), vals));
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
