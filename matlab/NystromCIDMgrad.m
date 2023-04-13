function [u,peq,qest,gradu, debug] = NystromCIDMgrad(X,KP)
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
    
    if (nargout>3) %%% compute the gradient of the squared distance
        v1 = repmat(X,[1 1 k])-permute(reshape(KP.X(:,inds),[n k N]),[1 3 2]); % shape [n N k]
        v1 = v1./repmat(sum(v1.^2)+eps,[n 1 1]); % shape [n N k]
        tmp = permute( ...
                reshape( ...
                    KP.X(:,inds(2:k2,:))', ...
                    [n k2-1 N] ...
                ), [1 3 2] ...
            ); % shape [n N k2-1]
        v2 = repmat(X,[1 1 k2-1]) - tmp;
        %tmp = v2;
        v2 = repmat(sum(v2,3),[1 1 k]); % shape [n N k]
        v2 = v2./repmat(sum(v2.^2)+eps,[n 1 1]); % shape [n N k]
        gradd = repmat(reshape(d',[1 N k]),[n 1 1]).*(2*v1 - v2); % shape [n N k]
    end
    
    
    %%% RBF Kernel
    d = exp(-d/(2*KP.epsilon));
    
    m=size(KP.lheat,1); %%% number of eigenfunctions
    
    if (nargout>3) %%% compute the gradients of each of the eigenfunctions
        gradu = sum(permute(repmat(d,[1 1 n]),[3 2 1]).*gradd,3)./repmat(sum(d),n,1);
        gradu = repmat(gradu,[1 1 k]) - gradd; % shape [n N k]
        uu = permute(repmat(reshape(KP.u(inds,:)',[m k N]),[1 1 1 n]),[4 3 2 1]); % shape [n N k m]
        gradu = squeeze(sum(uu.*permute(repmat(d,[1 1 n m]),[3 2 1 4]).* ...
            repmat(gradu,[1 1 1 m]),3))./repmat(sum(d),[n 1 m]); % shape [n N m]
        gradu = gradu./permute(repmat(KP.lheat,[1 n N]),[2 3 1]); %
    end
    
    d = sparse(reshape(double(inds),N*k,1),repmat(1:N,k,1),reshape(double(d),N*k,1),size(KP.X,2),N,N*k)';

    peq = full(sum(d,2));
    qest = peq./(rho.^KP.dim)'/N/((2*pi*KP.epsilon)^(KP.dim/2));
    
    D = spdiags(1./peq,0,N,N);
    d = D*d;
    
    
    Linv = spdiags(1./KP.lheat,0,m,m); %%% one over the eigenvalues, convert back to heat kernel eigs
    
    u = full(d*KP.u*Linv);

    if (nargout == 5)
        %debug = inds(2:k2,:);
        debug = tmp;
    end

end




