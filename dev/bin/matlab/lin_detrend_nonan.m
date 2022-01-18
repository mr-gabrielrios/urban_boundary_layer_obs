function [lindet] = lin_detrend_nonan(vel,fq,avg_window)

ptsperwin= fq*avg_window;
total_win= floor(length(vel)/ptsperwin);
lindet=zeros(ptsperwin*total_win,1);
totalwin1=int32(total_win);

m=0;

%fixed window 
for i=1:total_win
    istart=(i-1)*ptsperwin + 1;
    iend  =i*ptsperwin;
    %liner detrending using polyfit
    for j=istart:iend
        if isnan(vel(j))==1
            vel(j)=0;  
            m=m+1;
        end
    end

    p=polyfit((istart:iend)',vel(istart:iend),1);
    temp1_coeff1(i)=p(1,1);
    temp1_coeff2(i)=p(1,2);
    %averaging over the window
    avgd(i)=sum(vel(istart:iend))/(ptsperwin-m);

end

%sliding window
% 
% temp1_coeff1=ones(1,ptsperwin*total_win);
% temp1_coeff2=ones(1,ptsperwin*total_win);
% %p=ones(ptsperwin*total_win,2);
% for i=1:(ptsperwin*total_win)-ptsperwin+1
%     istart=i;
%     iend=i-1+ptsperwin;
%     i
%     %liner detrending using polyfit
%     p=polyfit(istart:iend,vel(istart:iend)',1);
%     temp1_coeff1(i)=p(1,1);
%     temp1_coeff2(i)=p(1,2);
% 
% %averaging over the window
%     %avgd(i)=sum(u(istart:iend))/ptsperwin;
%end

size(temp1_coeff1);

%loop to resize the coefficeint matrix
temp2_coeff1=ones(ptsperwin,total_win);
temp2_coeff2=ones(ptsperwin,total_win);

avgd1=ones(ptsperwin,total_win);

for j=1:total_win
    temp2_coeff1(1:ptsperwin,j)=temp1_coeff1(j);
    temp2_coeff2(1:ptsperwin,j)=temp1_coeff2(j);
    avgd1(1:ptsperwin,j) =avgd(j);
end
  
coeff1=reshape(temp2_coeff1,[ptsperwin*total_win 1]);
coeff2=reshape(temp2_coeff2,[ptsperwin*total_win 1]);

avgd2=reshape(avgd1,[ptsperwin*total_win 1]);

%determination of turbulent fluctuations
for k=1:ptsperwin*total_win
    lindet(k)=vel(k)-(coeff1(k)*k+coeff2(k));
end



    