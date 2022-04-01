% Modified by G. Rios on 2021/01/18

%Input the vector x and the sampling frequency Fs
function [Pxx,f] = powerspec_prathap(x,Fs)
% Get the next power of 2. Used to pad the FFT signal.
nfft = 2^nextpow2(length(x));
% Number of points that will be passed to the power spectra
% Only selects the positives
NumUniquePts = ceil((nfft+1)/2); 
% Generate frequency array
f = (0:NumUniquePts-1)*Fs/nfft; 
% Get power spectra
Pxx = abs(fft(x, nfft)).^2/length(x)/Fs;
% Truncate to the appropriate length
Pxx = 2.*Pxx(1:length(f));
% Get the running mean
Pxx = runmean(Pxx, 21);
