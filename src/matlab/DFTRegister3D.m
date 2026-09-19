function [params, regImg] = DFTRegister3D(fixedVolume, movingVolume, preFFT)
% Estimate integer translation by FFT cross-correlation of equal-sized volumes.
%
% fixedVolume and movingVolume are (row, column, Z) arrays. preFFT defaults to
% false; true means both inputs are already Fourier transformed. Returns params
% with shifts [rowShift, colShift, zShift] and diffphase. Optional regImg is the
% magnitude image from DFTApply3D. Pass params directly to DFTApply3D to correct
% the moving image; these are correction shifts, not Python displacement values.
if nargin < 3
    preFFT = false;
end

[nr,nc,nz] = size(movingVolume);
% ifftshift - Inverse zero-frequency shift
Nr = ifftshift(-fix(nr/2):ceil(nr/2)-1);
Nc = ifftshift(-fix(nc/2):ceil(nc/2)-1);
Nz = ifftshift(-fix(nz/2):ceil(nz/2)-1);

% debug
%size(fixedVolume)
%size(movingVolume)

if preFFT
    CC = ifftn(fixedVolume .* conj(movingVolume));
else
    CC = ifftn(fftn(fixedVolume) .* conj(fftn(movingVolume)));
end
CCabs = abs(CC);

[~,ix] = max(CCabs(:));
[i,j,k] = ind2sub(size(CCabs), ix);%ftr = fftFixed .* fftMoving;
CCmax = CC(i,j,k);
diffphase = angle(CCmax);

rowShift = Nr(i); 
colShift = Nc(j); 
zShift = Nz(k);

params = struct();
params.shifts = [rowShift, colShift, zShift];
params.diffphase = diffphase;

if nargout > 1   
    regImg = DFTApply3D(movingVolume, params, preFFT);            
end