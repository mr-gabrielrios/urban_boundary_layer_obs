function  [Sxx,f_vec,sigma_var,var1_for_fft_filt] = fft_xmin(var1_for_fft,var2_for_fft,leng,z_prime,tower_height,wspd)

    fs = 10; %10 hz sample rate
    dt = 1/fs; % time separation
    N = leng; % length in points
    T = N*dt; % total length in seconds
    df = 1/T; %frequency bins
    %
    threshold = 5; % amount greater than the standard deviation to throw away term
    %
    f_vec = (0:N/2).*df; % frequency vector for plotting
    
%     var_std_filt = find(var1_for_fft>5*nanstd(var1_for_fft) | var1_for_fft<-5*nanstd(var1_for_fft));
    var1_for_fft_filt = var1_for_fft;

    var1_for_fft_filt(abs(var1_for_fft)>((threshold*nanstd(var1_for_fft))+nanmean(var1_for_fft))) = nanmean(var1_for_fft);

%     
%     var_std_filt2 = find(var2_for_fft>5*nanstd(var2_for_fft) | var2_for_fft<-5*nanstd(var2_for_fft));
    var2_for_fft_filt = var2_for_fft;
    var2_for_fft_filt(abs(var2_for_fft)>((threshold*nanstd(var2_for_fft))+nanmean(var2_for_fft))) = nanmean(var2_for_fft);

%     if isempty(var_std_filt2)==1
%     else
%         var2_for_fft_filt(var_std_filt2) = nanmean(var2_for_fft);    
%     end
    
%     u_bar = mean(var1_for_fft);
    spec_1_xmin = fft(var1_for_fft_filt)/leng;
    sig_fft1 = spec_1_xmin(1:(N/2+1));
    sig_fft1(2:end-1) = 2.*sig_fft1(2:end-1);
    %
    spec_2_xmin = fft(var2_for_fft_filt)/leng;
    sig_fft2 = spec_2_xmin(1:(N/2+1));
    sig_fft2(2:end-1) = 2.*sig_fft2(2:end-1);
    %
    Sxx = sig_fft1.*conj(sig_fft2);
    sigma_var = sum(Sxx.*df);
%     scatter((f_vec.*z_prime)./wspd,(Sxx.*f_vec.')./sigma_var)
%     grid on
%     set(gca,'xscale','log','yscale','log','fontsize',24)
%     switch tower_height
%         
%         case 1
%             title('Normalized Power Spectra for Level 1')
%         case 2
%             title('Normalized Power Spectra for Level 2')
%         case 3
%             title('Normalized Power Spectra for Level 3')
%         case 4
%             title('Normalized Power Spectra for Level 4')
%         case 5
%             title('Normalized Power Spectra for Level 5')
%     end
%     xlabel('$nz''/ \overline{U}$','Interpreter','LaTex')
%     ylabel('$nP_{xx} /\sigma_x^2$','Interpreter','LaTex')

end