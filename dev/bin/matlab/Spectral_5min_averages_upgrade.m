%% Plotting Spectral Information 
%
clc
close all
% close all
%
if exist('Ts_1','var')==0
    clear
    uiopen('load')
else
end
%
%
%
%% Spectral calculation
%
% t_vec = linspace(0,100,100000);
% test_sig = sin(2*pi*2*t_vec);
%  
% var1 = [Ux_1 Uz_1];
% var2 = [Ux_2 Uz_2];
% var3 = [Ux_3 Uz_3];
% var4 = [Ux_4 Uz_4];
% var5 = [Ux_5 Uz_5];
%
for stab_desired = 4:-1:0; % neutral = 0, stable = 1, unstable = 2, very stable = 3, very unstable = 4

%
var1_0 = Ux_1;% Uz_1];
var2_0 = Ux_2;% Uz_2];
var3_0 = Ux_3;% Uz_3];
var4_0 = Ux_4;% Uz_4];
var5_0 = Ux_5;% Uz_5];
%
var11_0 = Uz_1;% Uz_1];
var22_0 = Uz_2;% Uz_2];
var33_0 = Uz_3;% Uz_3];
var44_0 = Uz_4;% Uz_4];
var55_0 = Uz_5;% Uz_5];

%
for pp = 1:length(var1_0(1,:))
    
    var1 = var1_0(:,pp);
    var2 = var2_0(:,pp);
    var3 = var3_0(:,pp);
    var4 = var4_0(:,pp);
    var5 = var5_0(:,pp);
    
leng = 3000; % 5 min or 5*60*10Hz
%
monin_param = .1;
monin_param2 = 1;
z = [5.9 12.1 17.1 23.5 35.9];
z_d = 3.2; % displacement height 3.2 +- 2.8;
z_prime = z-z_d;
%
set(0,'DefaultFigureWindowStyle','docked')
%
for ii = 1:floor(length(var1_0)/18000)
    var1_nan_test = mean(var1_0((ii-1)*18000+1:ii*18000));
    var2_nan_test = mean(var2_0((ii-1)*18000+1:ii*18000));
    var3_nan_test = mean(var3_0((ii-1)*18000+1:ii*18000));
    var4_nan_test = mean(var4_0((ii-1)*18000+1:ii*18000));
    var5_nan_test = mean(var5_0((ii-1)*18000+1:ii*18000));
%     
    var11_nan_test = mean(var11_0((ii-1)*18000+1:ii*18000));
    var22_nan_test = mean(var22_0((ii-1)*18000+1:ii*18000));
    var33_nan_test = mean(var33_0((ii-1)*18000+1:ii*18000));
    var44_nan_test = mean(var44_0((ii-1)*18000+1:ii*18000));
    var55_nan_test = mean(var55_0((ii-1)*18000+1:ii*18000));
%     
    [~,~,~,monin1,~,~,~,~,~,~,wspd1(ii),~,var1,dv1,var11,dt1] = turbstats_utes(Ux_1((ii-1)*18000+1:ii*18000),Uy_1((ii-1)*18000+1:ii*18000),Uz_1((ii-1)*18000+1:ii*18000),Ts_1((ii-1)*18000+1:ii*18000),time_stamp((ii-1)*18000+1:ii*18000));
    [~,~,~,monin2,~,~,~,~,~,~,wspd2(ii),~,var2,dv2,var22,dt2] = turbstats_utes(Ux_2((ii-1)*18000+1:ii*18000),Uy_2((ii-1)*18000+1:ii*18000),Uz_2((ii-1)*18000+1:ii*18000),Ts_2((ii-1)*18000+1:ii*18000),time_stamp((ii-1)*18000+1:ii*18000));
    [~,~,~,monin3,~,~,~,~,~,~,wspd3(ii),~,var3,dv3,var33,dt3] = turbstats_utes(Ux_3((ii-1)*18000+1:ii*18000),Uy_3((ii-1)*18000+1:ii*18000),Uz_3((ii-1)*18000+1:ii*18000),Ts_3((ii-1)*18000+1:ii*18000),time_stamp((ii-1)*18000+1:ii*18000));
    [~,~,~,monin4,~,~,~,~,~,~,wspd4(ii),~,var4,dv4,var44,dt4] = turbstats_utes(Ux_4((ii-1)*18000+1:ii*18000),Uy_4((ii-1)*18000+1:ii*18000),Uz_4((ii-1)*18000+1:ii*18000),Ts_4((ii-1)*18000+1:ii*18000),time_stamp((ii-1)*18000+1:ii*18000));
    [~,~,~,monin5,~,~,~,~,~,~,wspd5(ii),~,var5,dv5,var55,dt5] = turbstats_utes(Ux_5((ii-1)*18000+1:ii*18000),Uy_5((ii-1)*18000+1:ii*18000),Uz_5((ii-1)*18000+1:ii*18000),Ts_5((ii-1)*18000+1:ii*18000),time_stamp((ii-1)*18000+1:ii*18000));
%     var11 = var11_0;
%     var22 = var22_0;
%     var33 = var33_0;
%     var44 = var44_0;
%     var55 = var55_0;
    
    if isnan(var1_nan_test) || isnan(var11_nan_test)
        var1_for_fft(:,ii) = nan_fft(var1);
        var11_for_fft(:,ii) = nan_fft(var11);

    else
        var1_for_fft(:,ii) = var1;
        var11_for_fft(:,ii) = var11;
    end
%       
    [stability_param1] = stability_fft_single(1,monin1,monin_param,monin_param2);
    if stability_param1 == stab_desired;
        timestamp1_fft(ii) = time_stamp((ii-1)*18000+1);
        %figure
        for jj = 1:6
           [Sxx1_5min(:,jj),f_vec1,sigma_var1(:,jj),var1_for_fft_filt] = fft_xmin(var1_for_fft(((jj-1)*leng+1:jj*leng),ii),var11_for_fft(((jj-1)*leng+1:jj*leng),ii),leng,z_prime(1),1,wspd1(ii));           
           %hold all all
        end
    else
    end
    
    if isnan(var2_nan_test) || isnan(var22_nan_test)
        var2_for_fft(:,ii) = nan_fft(var2);
        var22_for_fft(:,ii) = nan_fft(var22);
    else
        var2_for_fft(:,ii) = var2;
        var22_for_fft(:,ii) = var22;
    end
    [stability_param2] = stability_fft_single(2,monin2,monin_param,monin_param2);
    if stability_param2 == stab_desired;
        timestamp2_fft(ii) = time_stamp((ii-1)*18000+1);
        %figure
        for jj = 1:6
           [Sxx2_5min(:,jj),f_vec2,sigma_var2(:,jj)] = fft_xmin(var2_for_fft(((jj-1)*leng+1:jj*leng),ii),var22_for_fft(((jj-1)*leng+1:jj*leng),ii),leng,z_prime(2),2,wspd2(ii));           
           %hold all all
        end
    else
    end
    
    if isnan(var3_nan_test) || isnan(var33_nan_test)
        var3_for_fft(:,ii) = nan_fft(var3);
        var33_for_fft(:,ii) = nan_fft(var33);
    else
        var3_for_fft(:,ii) = var3;
        var33_for_fft(:,ii) = var33;
    end
    [stability_param3] = stability_fft_single(3,monin3,monin_param,monin_param2);
    if stability_param3 == stab_desired;
        timestamp3_fft(ii) = time_stamp((ii-1)*18000+1);
        %figure
        for jj = 1:6
           [Sxx3_5min(:,jj),f_vec3,sigma_var3(:,jj)] = fft_xmin(var3_for_fft(((jj-1)*leng+1:jj*leng),ii),var33_for_fft(((jj-1)*leng+1:jj*leng),ii),leng,z_prime(3),3,wspd3(ii));           
           %hold all all
        end
    else
    end
    
    if isnan(var4_nan_test) || isnan(var44_nan_test)
        var4_for_fft(:,ii) = nan_fft(var4);
        var44_for_fft(:,ii) = nan_fft(var44);
    else
        var4_for_fft(:,ii) = var4;
        var44_for_fft(:,ii) = var44;
    end
    [stability_param4] = stability_fft_single(4,monin4,monin_param,monin_param2);
    if stability_param4 == stab_desired;
        timestamp4_fft(ii) = time_stamp((ii-1)*18000+1);
        %figure
        for jj = 1:6
           [Sxx4_5min(:,jj),f_vec4,sigma_var4(:,jj)] = fft_xmin(var4_for_fft(((jj-1)*leng+1:jj*leng),ii),var44_for_fft(((jj-1)*leng+1:jj*leng),ii),leng,z_prime(4),4,wspd4(ii));           
           %hold all all
        end
    else
    end
    
    if isnan(var5_nan_test) || isnan(var55_nan_test)
        var5_for_fft(:,ii) = nan_fft(var5);
        var55_for_fft(:,ii) = nan_fft(var55);
    else
        var5_for_fft(:,ii) = var5;
        var55_for_fft(:,ii) = var55;
    end
    [stability_param5] = stability_fft_single(5,monin5,monin_param,monin_param2);
    if stability_param5 == stab_desired;
        timestamp5_fft(ii) = time_stamp((ii-1)*18000+1);
        %figure
        for jj = 1:6
           [Sxx5_5min(:,jj),f_vec5,sigma_var5(:,jj)] = fft_xmin(var5_for_fft(((jj-1)*leng+1:jj*leng),ii),var55_for_fft(((jj-1)*leng+1:jj*leng),ii),leng,z_prime(5),5,wspd5(ii));           
           %hold all all
        end
    else
    end
        
    if stability_param1 == stab_desired
        Sxx1_cell{ii} = Sxx1_5min;
        sigma_var1_cell{ii} = sigma_var1;
        
    else
        Sxx1_cell{ii} = NaN;
        sigma_var1_cell{ii} = NaN;        
    end
    %
    if stability_param2 == stab_desired
        Sxx2_cell{ii} = Sxx2_5min;
        sigma_var2_cell{ii} = sigma_var2;        
    else
        Sxx2_cell{ii} = NaN;
        sigma_var2_cell{ii} = NaN;        
    end
    %
    if stability_param3 == stab_desired
        Sxx3_cell{ii} = Sxx3_5min;
        sigma_var3_cell{ii} = sigma_var3;        
    else
        Sxx3_cell{ii} = NaN;
        sigma_var3_cell{ii} = NaN;        
    end
    %
    if stability_param4 == stab_desired
        Sxx4_cell{ii} = Sxx4_5min;
        sigma_var4_cell{ii} = sigma_var4;        
    else
        Sxx4_cell{ii} = NaN;
        sigma_var4_cell{ii} = NaN;        
    end
    %
    if stability_param5 == stab_desired
        Sxx5_cell{ii} = Sxx5_5min;
        sigma_var5_cell{ii} = sigma_var5;        
    else
        Sxx5_cell{ii} = NaN;
        sigma_var5_cell{ii} = NaN;        
    end    
    stab_vec(ii,:) = [stability_param1 stability_param2 stability_param3 stability_param4 stability_param5];
end
%
%figures to keep

% Uncomment the following to 
% include ALL windows, including those with hidden handles (e.g. GUIs)
% all_figs = findall(0, 'type', 'figure');
%
Sxx1_cell_concat_for_mean = [];
Sxx2_cell_concat_for_mean = [];
Sxx3_cell_concat_for_mean = [];
Sxx4_cell_concat_for_mean = [];
Sxx5_cell_concat_for_mean = [];
%
sigma_var1_cell_concat_for_mean = [];
sigma_var2_cell_concat_for_mean = [];
sigma_var3_cell_concat_for_mean = [];
sigma_var4_cell_concat_for_mean = [];
sigma_var5_cell_concat_for_mean = [];
%
for kk = 1:length(Sxx1_cell)
    
    if isnan(Sxx1_cell{kk})
    else
        Sxx1_cell_concat_for_mean = [Sxx1_cell_concat_for_mean Sxx1_cell{kk}];
        sigma_var1_cell_concat_for_mean = [sigma_var1_cell_concat_for_mean sigma_var1_cell{kk}];        
    end
    
    if isnan(Sxx2_cell{kk})
    else
        Sxx2_cell_concat_for_mean = [Sxx2_cell_concat_for_mean Sxx2_cell{kk}];
        sigma_var2_cell_concat_for_mean = [sigma_var2_cell_concat_for_mean sigma_var2_cell{kk}];
    end
    
    if isnan(Sxx3_cell{kk})
    else
        Sxx3_cell_concat_for_mean = [Sxx3_cell_concat_for_mean Sxx3_cell{kk}];
        sigma_var3_cell_concat_for_mean = [sigma_var3_cell_concat_for_mean sigma_var3_cell{kk}];
    end
    
    if isnan(Sxx4_cell{kk})
    else
        Sxx4_cell_concat_for_mean = [Sxx4_cell_concat_for_mean Sxx4_cell{kk}];
        sigma_var4_cell_concat_for_mean = [sigma_var4_cell_concat_for_mean sigma_var4_cell{kk}];
    end
    
    if isnan(Sxx5_cell{kk})
    else
        Sxx5_cell_concat_for_mean = [Sxx5_cell_concat_for_mean Sxx5_cell{kk}];
        sigma_var5_cell_concat_for_mean = [sigma_var5_cell_concat_for_mean sigma_var5_cell{kk}];
    end    
    
end

for nn = 1:length (Sxx1_cell_concat_for_mean)    
    Sxx1_mean_collapse(nn) = mean(Sxx1_cell_concat_for_mean(nn,:));
end
if exist('fighand1')==0
%     fighand1 = figure;
end
wspd1_mean = mean(wspd1);
sigma_var1_mean = mean(sigma_var1_cell_concat_for_mean);
% print_title1 = strcat({'Normalized Power Spectra for '},'u During');%,{'-'});
% try
%     figure(fighand1)
% catch
    figure
% end
fft_xmin_concat(Sxx1_mean_collapse,wspd1_mean,sigma_var1_mean,f_vec1,z_prime(1),1,pp,stab_desired,time_stamp)
hold all
% h = legend('$u$','$w$');
% set(h,'Interpreter','LaTex')
% title(print_title1)
%
for nn = 1:length (Sxx2_cell_concat_for_mean)    
    Sxx2_mean_collapse(nn) = mean(Sxx2_cell_concat_for_mean(nn,:));
end
if exist('fighand2')==0
%     fighand2 = figure;
end
wspd2_mean = mean(wspd2);
sigma_var2_mean = mean(sigma_var2_cell_concat_for_mean);
% figure(fighand2)
s2 = fft_xmin_concat(Sxx2_mean_collapse,wspd2_mean,sigma_var2_mean,f_vec2,z_prime(2),2,pp,stab_desired,time_stamp);
hold all
% h = legend('$u$','$w$');
% set(h,'Interpreter','LaTex')
s2.Marker = 'hexagram';
s2.MarkerFaceColor = [255/256 150/256 255/256];
s2.MarkerEdgeColor = [102/256 0/256 102/256];
%
for nn = 1:length (Sxx3_cell_concat_for_mean)    
    Sxx3_mean_collapse(nn) = mean(Sxx3_cell_concat_for_mean(nn,:));
end
if exist('fighand3')==0
%     fighand3 = figure;
end
wspd3_mean = mean(wspd3);
sigma_var3_mean = mean(sigma_var3_cell_concat_for_mean);
% figure(fighand3)
s3 = fft_xmin_concat(Sxx3_mean_collapse,wspd3_mean,sigma_var3_mean,f_vec3,z_prime(3),3,pp,stab_desired,time_stamp);
hold all
% h = legend('$u$','$w$');
% set(h,'Interpreter','LaTex')
s3.Marker = 'o';
s3.MarkerFaceColor = [200/256 200/256 200/256];
s3.MarkerEdgeColor = [80/256 80/256 80/256];

%
for nn = 1:length (Sxx4_cell_concat_for_mean)    
    Sxx4_mean_collapse(nn) = mean(Sxx4_cell_concat_for_mean(nn,:));
end
if exist('fighand4')==0
%     fighand4 = figure;
end
wspd4_mean = mean(wspd4);
sigma_var4_mean = mean(sigma_var4_cell_concat_for_mean);
% figure(fighand4)
s4 = fft_xmin_concat(Sxx4_mean_collapse,wspd4_mean,sigma_var4_mean,f_vec4,z_prime(4),4,pp,stab_desired,time_stamp);
hold all
% h = legend('$u$','$w$');
% set(h,'Interpreter','LaTex')
s4.Marker = 'square';
s4.MarkerFaceColor = [186/256 212/256 244/256];
s4.MarkerEdgeColor = [52/256 77/256 126/256];
%
for nn = 1:length (Sxx5_cell_concat_for_mean)    
    Sxx5_mean_collapse(nn) = mean(Sxx5_cell_concat_for_mean(nn,:));
end
if exist('fighand5')==0
%     fighand5 = figure;
end
wspd5_mean = mean(wspd5);
sigma_var5_mean = mean(sigma_var5_cell_concat_for_mean);
% figure(fighand5)
s5 = fft_xmin_concat(Sxx5_mean_collapse,wspd5_mean,sigma_var5_mean,f_vec5,z_prime(5),5,pp,stab_desired,time_stamp);
hold all
h = legend('$1$','$2$','$3$','$4$','$5$');
set(h,'Interpreter','LaTex')
s5.Marker = 'pentagram';
s5.MarkerFaceColor = [0/256 250/256 0/256];
s5.MarkerEdgeColor = [0/256 127/256 0/256];
ylabel('$nP_{xy} /\sigma_x \sigma_y$','Interpreter','LaTex')
%%
f_vec1_plot = ((f_vec1.*z_prime(1))./wspd1_mean).^-(2/3);
%
plot((f_vec1.*z_prime(1))./wspd1_mean,f_vec1_plot/10,'linewidth',4)
%%


clearvars -except Ux_1 Ux_2 Ux_3 Ux_4 Ux_5 Uy_1 Uy_2 Uy_3 Uy_4 Uy_5 Uz_1...
    Uz_2 Uz_3 Uz_4 Uz_5 Ts_1 Ts_2 Ts_3 Ts_4 Ts_5 time_stamp...
    fighand1 fighand2 fighand3 fighand4 fighand5 monin1 monin2 monin3 monin4 monin5
    
end

end
    

