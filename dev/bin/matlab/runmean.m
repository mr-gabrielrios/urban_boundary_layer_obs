% Hi Per,
% The function below performs a running mean, such that Y(i) = mean(X(i-m/2:i+m/2). Check if
% the behavior for odd and even m's and at the ends of the array X satisfies your needs. It is
% designed for real vectors but it is very fast!
% hth Jos
% 
% ========================================
function y = runmean(X,m) ;
% RUNMEAN
% Incredibly fast Running Mean filter, since execution time does not depend on
% the size of the filter window!
% Y = RUNMEAN(X,M) computes a running mean on vector X using a window of M datapoints.
% X is a vector, and M is a positive integer defining the size of the window.
%
% N.B. length(Y) = length(X) ;

error(nargchk(2,2,nargin))
if (isempty(X)|isempty(m))
   y = [];
   return
elseif m==1,
   y = X ;
   return ;
elseif (prod(size(m)) ~= 1) | (m < 2),
   error('m should be a positive integer') ;
end
[nrow,ncol] = size(X) ;
if (ncol>1)&(nrow>1)
   error('X should be a vector') ;
end
% This is the trick !!
% --------------------
x1 = repmat(X(1),ceil((m-1)/2),1) ;
x2 = repmat(X(end),floor((m-1)/2),1) ;
X = [x1 ; X(:) ; x2] ;
z = [0 ; cumsum(X)] ;
y = (z(m+1:end)-z(1:end-m)) / m ;

if nrow==1,
   y = y' ;   % convert column to row if row-input
end
