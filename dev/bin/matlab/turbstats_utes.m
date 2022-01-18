function [ustar,ustar_uw_av,tke,monin,sigmau,sigmav,sigmaw,sigmat,tstar,H,wspd,timestamp,du,dv,dw,dt] = turbstats_utes(u,v,w,t,timestamp)

%Filtering csat3 data
% u(find(csatwarnings>0))=NaN;v(find(csatwarnings>0))=NaN;
% w(find(csatwarnings>0))=NaN;t(find(csatwarnings>0))=NaN;

timestamp = timestamp(1:18000:end);

u = nan_5std_dev_rep_30min(u);  %replace NaN and > 5std with the block mean
u(find(diff(u)>10)) = mean(u);
v = nan_5std_dev_rep_30min(v);  %replace NaN and > 5std with the block mean
v(find(diff(v)>10)) = mean(v);
w = nan_5std_dev_rep_30min(w);  %replace NaN and > 5std with the block mean
w(find(diff(w)>10)) = mean(w);
t = nan_5std_dev_rep_30min(t);  %replace NaN and > 5std with the block mean
t(find(diff(t)>10)) = mean(t);


%Despiking, the second input variable(5) determines how many points to
%check. A value of 3 identifies single point spikes
% t=despike(t,3,2);
% abc=find(t==NaN);
% u(abc)=NaN;v(abc)=NaN;w(abc)=NaN;
% c=despike(c,3,2);
% abc2=find(c==NaN);
% q(abc2)=NaN;

%Applying AGC filter
%  ANS1=repmat(AGC,1,18000);
%  ANS2=reshape(ANS1',length(AGC)*18000,1);
 %q(find(ANS2>56))=NaN;c(find(ANS2>56))=NaN;
 wspd=nan_average_30(sqrt(u.^2+v.^2));
 
%performing planar rotation
[u_r,v_r,w_r] = wilczak_2016(u,v,w,length(nan_average_30(u)));


du=lin_detrend_nonan(u_r,10,1800);
dv=lin_detrend_nonan(v_r,10,1800);
dw=lin_detrend_nonan(w_r,10,1800);
dt=lin_detrend_nonan(t,10,1800);
% dc=lin_detrend_nonan(c,10,1800);
% dq=lin_detrend_nonan(q,10,1800);

%calculating ustar, tke
ustar_uw=(du.*dw);ustar_vw=(dv.*dw);ustar_uw_av=nan_average_30(ustar_uw);ustar_vw_av=nan_average_30(ustar_vw);
ustar=(ustar_uw_av.^2+ustar_vw_av.^2).^.25;
tke = 0.5.*(nan_average_30(du.^2)+nan_average_30(dv.^2)+nan_average_30(dw.^2));

%calculating standard deviations
sigmau=nan_std_30(u);
sigmav=nan_std_30(v);
sigmaw=nan_std_30(w);
sigmat=nan_std_30(t);
% sigmac=nan_std_30(c);
% sigmaq=nan_std_30(q);

%calculating fluxes
%declaring constants
Rw  = 461.5;     % ideal gas constant for water vapor, J/kg*K
Rd  = 287.05;    % ideal gas constant for dry air, J/kg*K
Lv  = 1000*2260; % latent heat of vaporization (water), J/kg
Cp  = 1005;      % approximate constant pressure specific heat of air, J/kg*K
k   = 0.4;       % Von Karman constant
g   = 9.8;      




% T     = Tair + 273.15;    

rho_a = 1.2;%(Press.*1000)./(Rd.*T);

% Fc=nan_average_30(dw.*dc(1:length(dw)));E=nan_average_30(dw.*dq(1:length(dw)));
wT=nan_average_30(dw.*dt(1:length(dw)));

% rho_v_avg=nan_average_30(q);rho_c_avg=nan_average_30(c);
% rho_v_avg=rho_v_avg(1:length(Fc));rho_c_avg=rho_c_avg(1:length(Fc));
% uw=ustar_uw_av;

%  mu     = 28.966/18.016;                     % ratio of molar masses of water vapor and air
% sigma  = rho_v_avg./(nan_average_30(rho_a)-rho_v_avg);      % mixing ratio

% k_webb = 0.5.*nan_average_30(rho_a).*nan_average_30(u).^2./(nan_average_30(Press).*1000);        % pressure coefficient

% calculate Webb-corrected average vertical velocity
% w_webb = (1+mu*sigma+k_webb).*wT./nan_average_30(T) + mu*sigma.*E./rho_v_avg...
%     + 2*k_webb.*uw./nan_average_30(u);
% w_avg  = w_webb;

% E  = E  + w_webb.*rho_v_avg;                % zhihuaw
% Fc_1 = Fc + w_webb.*rho_c_avg;                % zhihuaw

% LE = Lv*E;                                      % latent heat flux, W/m^2
H  = Cp*rho_a.*wT;                              % sensible heat flux, W/m^2
% Fc        = Fc_1.*1e6;  




tstar=-nan_average_30(dw.*dt)./ustar;
% cstar=-(Fc)./ustar;
% qstar=-(E)./ustar;

monin=-ustar.^3.*(nan_average_30(t))./(.4.*9.8.*nan_average_30(dw.*dt));

