function cs = Str2Colorseq( s )
% Convert a character vector such as '1234' into numeric color labels.
%
% s must be a character vector: the implementation loops over numel(s) and
% uses str2num on each character. Returns a numeric row vector cs. Do not pass
% a scalar MATLAB string expecting per-character iteration.
cs = [];
for i=1:numel(s)
    cs = [cs str2num(s(i))];
end

end

