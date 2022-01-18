%Input the vectors, x and y and the sampling frequency  Fs 
function [Cxy,f]= cospec_prathap(x,y,Fs)
   


nfft = 2^nextpow2(length(x));
NumUniquePts = ceil((nfft+1)/2); 
f = (0:NumUniquePts-1)*Fs/nfft; 

Pxx = (fft(x,nfft)).^2/nfft/Fs;Pyy = (fft(y,nfft)).^2/nfft/Fs;


Cxy=real(Pxx).*real(Pyy)+imag(Pxx).*imag(Pyy);
Cxy=2.*Cxy(1:length(f));
Cxy=runmean(Cxy,21);