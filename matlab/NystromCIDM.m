function [u,peq,qest] = NystromCIDM(X,KP)
%%% Conformally Invariant Diffusion Map (CIDM)
%%% Inputs
    %%% x       - n-by-N data set with N data points in R^n
    %%% KP      - data structure output from CIDM that stores the map info
    
%%% Outputs
    %%% u       - Eigenfunctions of the generator/Laplacian extended to x
    %%% peq     - Invariant measure of diffusion process exteded to x
    %%% qest    - Sampling measure extended to x

    [n,N]=size(X);
    k=KP.k;
    k2=KP.k2;
    
    [d,inds]=pdist2(KP.X',X','euclidean','smallest',k);

    %%% CkNN Normalization
    rho=mean(d(2:k2,:));
    d = (d.^2)./(repmat(rho,k,1).*KP.rho(inds));
    
    %%% RBF Kernel
    d = exp(-d/(2*KP.epsilon));
    d = sparse(reshape(double(inds),N*k,1),repmat(1:N,k,1),reshape(double(d),N*k,1),size(KP.X,2),N,N*k)';

    peq = full(sum(d,2));
    qest = peq./(rho.^KP.dim)'/N/((2*pi*KP.epsilon)^(KP.dim/2));
    
    D = spdiags(1./peq,0,N,N);
    d = D*d;
    
    nvar=size(KP.lheat,1);
    Linv = spdiags(1./KP.lheat,0,nvar,nvar); %%% one over the eigenvalues, convert back to heat kernel eigs
    
    u = full(d*KP.u*Linv);
    
end




