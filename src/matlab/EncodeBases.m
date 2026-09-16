function colorSeq = EncodeBases( seq )
% Encode a scalar DNA string of length L into L-1 color characters.
%
% seq uses uppercase A/C/G/T. Adjacent pairs map to labels 1 through 4; the
% returned colorSeq is a scalar string. AA/CC/GG/TT map to 1; AC/CA/GT/TG to 2;
% AG/CT/GA/TC to 3; AT/CG/GC/TA to 4. Unknown pairs are not accepted.

    % construct hash table for encoding
    % k = {'AT','CT','GT','TT',...
    %     'AG', 'CG', 'GG', 'TG',...
    %     'AC', 'CC', 'GC', 'TC',...
    %     'AA', 'CA', 'GA', 'TA'};
    % v = {4,3,2,1,...
    %     3,4,1,2,...
    %     2,1,4,3,...
    %     1,2,3,4};
    % coding = containers.Map(k,v);
   
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

    
    start = 1;
    back = start + 1;
    colorSeq = "";
    while back <= strlength(seq)
        curr_str = extractBetween(seq, start, back);
        % colorSeq = colorSeq + coding(curr_str);
        colorSeq = colorSeq + base_to_color(curr_str);
        start = start + 1;
        back = start + 1;
    end
    
end

