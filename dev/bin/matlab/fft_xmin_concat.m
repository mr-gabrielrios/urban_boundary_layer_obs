function  [s] = fft_xmin_concat(var1_fft,wspd_mean,sigma_var_mean,f_vec,z_prime,tower_height,pp,stab_desired,time_stamp)

   
    s = scatter((f_vec.*z_prime)./wspd_mean,(var1_fft.*f_vec)./sigma_var_mean);
    grid on
    set(gca,'xscale','log','yscale','log','fontsize',24)
%     title_str_height = sprintf('Normalized Power Spectra for Level %d',tower_height);
    title_str_height = 'Normalized Power Spectra for';
    title_str_height = strcat(title_str_height,' u''w''');
    switch stab_desired
        case 0
            title_str_stab = strcat(title_str_height,' [Neutral ');           
        case 1
            title_str_stab = strcat(title_str_height,' [Stable '); 
        case 2
            title_str_stab = strcat(title_str_height,' [Unstable '); 
        case 3
            title_str_stab = strcat(title_str_height,' [Very Stable '); 
        case 4
            title_str_stab = strcat(title_str_height,' [Very Unstable ');             
    end
    title_str = strcat(title_str_stab,{' '},time_stamp{1}(6:10),{' to '},time_stamp{end}(6:10),']');
    title(title_str)
    xlabel('$nz''/ \overline{U}$','Interpreter','LaTex')
    ylabel('$nS_{xy} /\overline{u''w''}$','Interpreter','LaTex')
    xlim([0.001 100])
    ylim([0.001 1])
    if pp==1
        s.Marker = 'diamond';
        s.MarkerFaceColor = [162/256 20/256 47/256];
        s.MarkerEdgeColor = [135/256 81/256 81/256];
    elseif pp==2
        s.Marker = 'hexagram';
        s.MarkerFaceColor = [119/256 172/256 48/256];
        s.MarkerEdgeColor = [59/256 113/256 86/256];
        
    else
    end
        

end