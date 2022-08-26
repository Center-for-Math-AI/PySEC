function [U,L,D1,G,cijk] = Del1(u,l,D,n1,n2)
%%% u = N-by-n0 matrix of eigenfunctions of the 0-Laplacian
%%% l = n0-by-1 vector of eigenvalues of the 0-Laplacian (positive definite)
%%% D = inner product, diagonal matrix such that u'*D*u = I
%%% n1 = number of eigenfunctions to use as functions,b^i db^j, i=1,...,n1
%%% n2 = number of eigenfunctions to use as forms, j=1,...,n2
%%% n0 = size(u,2) = number of eigenfunctions for expanding products: b^i*b^j = sum_{k=1}^n0 c_{ijk} b^k

    [~,n0]=size(u);
    
    if (nargin<4)   n1 = n0;    end
    if (nargin<5)   n2 = n1;    end
    
    %%% Vectorized computation of the cijk 3-tensor where
    %%% cijk = <phi_i * phi_j, phi_k> = (u(:,i).*u(:,j))'*D*u(:,k)
    %%% are the structure constants for the multiplicative algebra
    cijk = repmat(u(:,1:n1),[1 1 n1 n0]);
    cijk = squeeze(mean(cijk.*permute(cijk,[1 3 2 4]).*permute(repmat(D*u,[1 1 n1 n1]),[1 3 4 2])));
    
    l1 = permute(repmat(l,[1 n1 n1 n2 n2]),[2 4 1 3 5]);
    
    cijkl = squeeze(sum(repmat(cijk,[1 1 1 n2 n2]).*permute(repmat(cijk(1:n2,1:n2,:),[1 1 1 n1 n1]),[4 5 3 1 2]),3));
    cikjls = squeeze(sum(permute(repmat(cijk,[1 1 1 n2 n2]),[1 4 3 2 5]).*l1.*permute(repmat(cijk(1:n2,1:n2,:),[1 1 1 n1 n1]),[4 1 3 5 2]),3));
    
    l2 = repmat(l(1:n1),[1 n1 n2 n2]);
    l3 = repmat(l(1:n2),[1 n2 n1 n1]);
    G = (1/2)*((permute(l3,[3 1 4 2]) + permute(l3,[3 2 4 1])).*permute(cijkl,[1 3 2 4]) - cikjls);
    
    clear cijkl;

    ciljks = squeeze(sum(permute(repmat(cijk(:,1:n2,:),[1 1 1 n1 n2]),[1 5 3 4 2]).*l1.*permute(repmat(cijk(:,1:n2,:),[1 1 1 n1 n2]),[4 2 3 1 5]),3));
    
    D1 = (1/4)*(permute(l2,[1 3 2 4]) + permute(l3,[3 1 4 2]) + permute(l2,[2 3 1 4]) + permute(l3,[3 2 4 1])).*(ciljks-cikjls);
    
    clear ciljks;
    
    cijkls = squeeze(sum(repmat(cijk(:,1:n2,:),[1 1 1 n1 n2]).*l1.*permute(repmat(cijk(:,1:n2,:),[1 1 1 n1 n2]),[4 5 3 1 2]),3));
    
    D1 = D1 + (1/4)*(-permute(l2,[1 3 2 4]) + permute(l3,[3 1 4 2]) - permute(l2,[2 3 1 4]) + permute(l3,[3 2 4 1])).*cijkls;
    
    clear cijkls l2 l3;
    
    cikjls2 = squeeze(sum(permute(repmat(cijk,[1 1 1 n2 n2]),[1 4 3 2 5]).*l1.^2.*permute(repmat(cijk(1:n2,1:n2,:),[1 1 1 n1 n1]),[4 1 3 5 2]),3));
    
    D1 = D1 + (1/4)*cikjls2;
    
    clear cikjls2;
    
    cijkls2 = squeeze(sum(repmat(cijk(:,1:n2,:),[1 1 1 n1 n2]).*l1.^2.*permute(repmat(cijk(:,1:n2,:),[1 1 1 n1 n2]),[4 5 3 1 2]),3));
    
    D1 = D1 + (1/4)*cijkls2;
    
    clear cijkls2;
    
    ciljks2 = squeeze(sum(permute(repmat(cijk(:,1:n2,:),[1 1 1 n1 n2]),[1 5 3 4 2]).*l1.^2.*permute(repmat(cijk(:,1:n2,:),[1 1 1 n1 n2]),[4 2 3 1 5]),3));
    
    D1 = D1 - (1/4)*ciljks2;
    
    clear ciljks2 l1;

    %%%%%%%%%%%%%%%%
    
    %%% The 4-dimensional tensor can now be reshaped into symmetric
    %%% matrices
    D1 = reshape(D1,n1*n2,n1*n2);
    G = reshape(G,n1*n2,n1*n2);
    
    D1 = (D1+D1')/2;
    G = (G+G')/2;
    
    %%% D1 is the energy matrix for the 1-Laplacian and G is the Hodge
    %%% Grammian matrix.  The sum D1+G is the Sobolev H^1 Grammian.  This
    %%% SVD computes the frame representations of a basis for H^1.
    [Ut,St,~]=svd(D1+G);
    St = diag(St);
    NN = find(St/St(1,1) < 1e-3,1);
    
    %%% We then project the 1-Laplacian and Hodge Grammian into the H^1
    %%% basis and insure that the results are symmetric.
    D1proj = Ut(:,1:NN)'*D1*Ut(:,1:NN);
    D1proj = (D1proj+D1proj')/2;
    Gproj = Ut(:,1:NN)'*G*Ut(:,1:NN);
    Gproj = (Gproj+Gproj')/2;
    
    %%% We can now compute the eigenforms of the 1-Laplacian in this basis
    [U,L] = eig(D1proj,Gproj);
    [L,sinds]=sort(abs(diag(L)));
    U = real(U(:,sinds));
    %%% Finally we compute the frame coefficients of the eigenforms
    U = Ut(:,1:NN)*U;
   
    
    

    
