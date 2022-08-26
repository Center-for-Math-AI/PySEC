function [u,l,peq,qest,epsilon,dim,KP] = CIDM(X,nvars,k,k2,tuningMethod)
%%% Conformally Invariant Diffusion Map (CIDM)
%%% Inputs
    %%% x       - n-by-N data set with N data points in R^n
    %%% nvars   - number of eigenfunctions/eigenvalues to compute
    %%% k       - number of nearest neighbors to use
    %%% k2      - number of nearest neighbors to use to determine the "epsilon"
    %%%             parameter
    %%% tuningMethod - methods of autotuning epsilon
    
%%% Outputs
    %%% u       - Eigenfunctions of the generator/Laplacian
    %%% l       - Eigenvalues
    %%% peq     - Invariant measure of the diffusion process, u'*diag(peq)*u = Id, so u are orthogonal wrt peq
    %%% qest    - Sampling measure
    %%% epsilon - scale, derived from the tuning kernel sum
    %%% dim     - intrinsic dimension, derived from the tuning kernel sum
    %%% KP      - Kernel parameters used for NystromCIDM extension

    [n,N]=size(X);
    
    if (nargin<5)   tuningMethod = 0;   end
    if (nargin<4)   k2=ceil(log(N));    end
    if (nargin<3)   k=ceil(log(N)^2);   end
    if (nargin<2)   nvars = 2*k;        end

    [d,inds] = pdist2(X',X','euclidean','smallest',k);

    %%% Bandwidth function, proportional to q^(-1/dim)
    rho = mean(d(2:k2,:));

    %%% CkNN/CIDM normalized distances
    d = d.^2./(repmat(rho,k,1).*rho(inds));

    %%% Tune epsilon and estimate dimension
    if (tuningMethod == 0)  
        epsilon = mean(mean(d(2:k2,:)));
    else
        epsilon = tuneEpsilon(d,tuningMethod);
    end
    [dim, ddim] = estimateDimension(d,epsilon);

    %%% Exponential kernel
    d = exp(-d./(2*epsilon));
    d = sparse(reshape(double(inds),N*k,1),repmat(1:N,k,1),reshape(double(d),N*k,1),N,N,N*k)';
    KP.d=d;
    %d = max(d, d'); % handle equidistant points by effectily upping k by 1 in that case
    %d = sqrt(d.^2 + d'.^2 - d.*d'); % handle equidistant points and symmetrize
    d = (d+d')/2;

    %%% CIDM Normalization
    peq = full(sum(d,2));
    Dinv = spdiags(peq.^(-1/2),0,N,N);
    d = Dinv*d*Dinv;
    d = (d+d')/2;

    if (nvars > 0)
    
        [u,l] = eigs(d,nvars,1.01);
        %[u,l] = eigs(full(d),nvars,1.1);

        l = (diag(l));
        [~,perm] = sort(abs(l),'descend');
        KP.lheat = l;                        %%% save the heat kernel eigs
        l = abs(log(l(perm)))/epsilon;       %%% convert to Laplacian eigs
        u = Dinv*u(:,perm);
        
    end

    qest = peq./(rho.^dim)'/N/((2*pi*epsilon)^(dim/2));
    
    KP.rho=rho;
    KP.epsilon=epsilon;
    KP.dim=dim;
    KP.k=k;
    KP.k2=k2;
    KP.X=X;
    KP.peq=peq;
    KP.u=u;
    KP.l=l;

end




