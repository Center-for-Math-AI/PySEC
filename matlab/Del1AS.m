function [U,L,D1,G,H,cijk] = Del1AS(u,l,D,n)
%%% Construction of the 1-Laplacian in the anti-symmetric frame

%%% u = N-by-n0 matrix of eigenfunctions of the 0-Laplacian
%%% l = n0-by-1 vector of eigenvalues of the 0-Laplacian (positive definite)
%%% D = inner product, diagonal matrix such that u'*D*u = I
%%% n = number of eigenfunctions to use as frame for 1-forms: b^{ij} = b^i db^j - b^j db^i, i=1,...,n1
%%% Note: n0 = size(u,2) = number of eigenfunctions for expanding products: b^i*b^j = sum_{k=1}^n0 c_{ijk} b^k
%%% U = Eigenforms of the 1-Laplacian, (frame coefficients)
%%% L = Eigenvalues of the 1-Laplacian
%%% D1= Matrix representation of the 1-Laplacian (in the frame): E_{ijkl} = <b^{ij},\Delta_1(b^{kl})>
%%% G = Hodge Grammian for the frame elements: G_{ijkl} = <b^{ij},b^{kl}>
%%% H = Operator represntation of frame elements, H_{ijkl} =
%%%     <b^l,b^{ij}(b^k)>. H converts between frame and operator representations
%%% cijk = Representation of pointwise multiplication in the Fourier basis
    
    n0=size(u,2);
    
    if (nargin<3) 
        n = n0; 
    end
    
    cijk = repmat(u(:,1:n),[1 1 n n0]);
    cijk = squeeze(sum(cijk.*permute(cijk,[1 3 2 4]).*permute(repmat(D*u,[1 1 n n]),[1 3 4 2])));

    l1 = repmat(l,[1 n n n n]);
    l2 = repmat(l(1:n),[1 n n0 n n]);
    h0 = repmat(cijk,[1 1 1 n n]);
    h0 = squeeze(sum(permute(h0,[4 1 3 5 2]).*permute(h0,[1 4 3 2 5]).*(permute(l2,[2 1 3 4 5]) + permute(l2,[4 2 3 5 1]) - permute(l1,[3 2 1 5 4])),3));

    H = h0 - permute(h0,[2 1 3 4]);
    G = h0 + permute(h0,[2 1 4 3]) - permute(h0,[1 2 4 3]) - permute(h0,[2 1 3 4]);
    
    clear h0;
    
    D1 = repmat(cijk,[1 1 1 n n]);   
    lambdas1 = permute(l1,[2 3 1 4 5]).^2 - permute(l1,[2 3 1 4 5]).*(l2 + permute(l2,[2 1 3 4 5])+permute(l2,[2 4 3 1 5])+permute(l2,[2 4 3 5 1]));
    D1 = 2*squeeze(sum((permute(D1,[2 5 3 1 4]).*permute(D1,[5 2 3 4 1])-permute(D1,[5 2 3 1 4]).*permute(D1,[2 5 3 4 1])).*lambdas1,3));

    %%%%%%%%%%%%%%%%
    
    %%% The 4-dimensional tensor can now be reshaped into symmetric
    %%% matrices
    D1 = reshape(D1,n^2,n^2);
    G = reshape(G,n^2,n^2);
    H = reshape(H,n^2,n^2);
    
    D1 = (D1+D1')/2;
    G = (G+G')/2;
    
    %%% D1 is the energy matrix for the 1-Laplacian and G is the Hodge
    %%% Grammian matrix.  The sum D1+G is the Sobolev H^1 Grammian.  This
    %%% SVD computes the frame representations of a basis for H^1.
    [Ut,St,~]=svd(D1+G);
    St = diag(St);
    
    %%% We truncate the basis to remove the very high H^1 energy forms
    %%% which are spurious due to the redundant frame representation (D1
    %%% and G should be rank deficient due to redundancy).
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

    

    
