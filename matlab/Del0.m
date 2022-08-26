function [u,l,D] = Del0(x,n,epsilon)
%%% Construction of the 0-Laplacian using the Diffusion Maps algorithm
%%% x = m-times-N matrix representing N data points in R^m
%%% n = number of eigenfunctions to compute
%%% epsilon = optional bandwidth parameter
%%% u: Eigenfunctions u(i,n) = phi_n(x_i)
%%% l: Eigenvalues l(n) = lambda_n
%%% D: inner product (diagonal matrix), u'*D*u = I.

    d = pdist2(x',x');
  
    if (nargin<3)
        k=1+ceil(log(n));
        epsilon = mink(d,k);
        epsilon = mean(epsilon(k,:))
    end

    d = exp(-d.^2/epsilon^2/4);
    
    d = (d+d')/2;
    
    D = diag(1./sum(d,2));
    d = D*d*D;
    
    D = diag(sum(d,2));
    d = (d + d')/2;
    
    [u,l] = eigs(d,D,n,'LM');
    l=diag(l);
    [l,sinds]=sort(real(l),'descend');
    l=-log(l)/epsilon^2;
    u=u(:,sinds);
    
    