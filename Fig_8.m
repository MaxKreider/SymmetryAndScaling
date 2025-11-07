%% setup

%mr clean
clc
clf
clear
close all

%choose data type fixed equal spacing, or random
data_type = 2;

%number of data points to consider ^2
N = 70;

%number of nearest neighbors to consider
K = 40;

%degree of local polynomial fit
ell = 4;

%number of GMLS iterations to perform
J = 2;

%domain
a = 1;
b = 2;


%% step 1 - generate data: u(t,x) = t + x

%generate data
if data_type == 1

    %fixed equally spaced values
    t = linspace(a,b,N);
    [t,x] = meshgrid(t,t);
    X(:,1) = t(:);
    X(:,2) = x(:);

else

    %uniformly distributed random samples in [a,b]^2
    X = a + (b - a) * rand(N^2, 2);

end

%include u
X = [X, 1./sqrt(4*pi*X(:,1)).*exp(-X(:,2).^2./(4*X(:,1)))];


%% step 2 - estimate tangent and normal vectors via SVD (rough approximation)

%keep track of normal directions
S = zeros(3,N);
T1 = zeros(3,N);
T2 = zeros(3,N);

%nearest neighbors
[idx_all, ~] = knnsearch(X, X, 'K', K);

%for each data point x_i, we will do the following:
for i=1:size(X,1)

    %find the K nearest neighbors of x_i and store them in neighbor
    index = idx_all(i,:);
    neighbor = X(index,:);

    %construct the matrix M_i
    M = neighbor' - X(i,:)';

    %take the SVD of M
    [U,~,~] = svd(M);

    %track the left singular vector corresponding to largest sv (tangent vector approximation)
    T1(:,i) = U(:,1);
    T2(:,i) = U(:,2);

    %find normal vector
    S_temp = null([T1(:,i),T2(:,i)]');
    S(:,i) = S_temp/norm(S_temp);


    %% step 3 - implement GMLS to improve the SVD approximation

    %iterate GMLS solver
    for j=1:J

        %project the differences onto tangent,normal plane for xi
        x_proj = (M'*[T1(:,i),T2(:,i),S(:,i)])';

        %form a vandermonde-like matrix
        V = [];
        for d = 1:ell
            for pp = 0:d
                qq = d - pp;
                if pp==0 && qq==0
                    continue
                end
                V = [V, ((x_proj(1,2:end).^pp).*(x_proj(2,2:end).^qq))'];
            end
        end

        %solve for the coefficient
        coeff = (V\(x_proj(3,2:end))');
        a1 = coeff(1:2);

        %update tangent vector
        t_new1 = T1(:,i) + a1(2)*S(:,i);
        t_new2 = T2(:,i) + a1(1)*S(:,i);
        t_new1 = t_new1/norm(t_new1);
        t_new2 = t_new2/norm(t_new2);

        T1(:,i) = t_new1;
        T2(:,i) = t_new2;

        %update normal vector
        s_new = null([T1(:,i),T2(:,i)]');
        S(:,i) = s_new(:,1)/norm(s_new(:,1));

    end
end


%% step 4 - compute the derivatives

%solve a linear system for du
du = zeros(2,size(X,1));
for pp=1:size(X,1)
    A = [T1(1:2,pp),T2(1:2,pp)]';
    b = [T1(3,pp); T2(3,pp)];
    du(:,pp) = A\b;
    cond1(pp) = cond(A);
end

%ratio of tangent vector to compute x_t and y_t
X = [X, du(1,:)', du(2,:)'];


%% step 5 - estimate tangent and normal vectors of jet space via SVD (rough approximation)

%keep track of normal directions
S1 = zeros(5,size(X,1));
S2 = zeros(5,size(X,1));
S3 = zeros(5,size(X,1));
T_v2_1 = zeros(5,size(X,1));
T_v2_2 = zeros(5,size(X,1));

%nearest neighbors
[idx_all, ~] = knnsearch(X, X, 'K', K);

%for each data point x_i, we will do the following:
for i=1:size(X,1)

    %find the K nearest neighbors of x_i and store them in neighbor
    index = idx_all(i,:);
    neighbor = X(index,:);

    %construct the matrix M_i
    M = neighbor' - X(i,:)';

    %take the SVD of M
    [U,~,~] = svd(M);

    %track the left singular vector corresponding to largest sv (tangent vector approximation)
    T_v2_1(:,i) = U(:,1);
    T_v2_2(:,i) = U(:,2);

    %do it for the normal vectors
    S_temp = null([T_v2_1(:,i),T_v2_2(:,i)]');
    S1(:,i) = S_temp(:,1)/norm(S_temp(:,1));
    S2(:,i) = S_temp(:,2)/norm(S_temp(:,2));
    S3(:,i) = S_temp(:,3)/norm(S_temp(:,3));


    %% step 6 - implement GMLS to improve the SVD approximation in the jet space

    %iterate GMLS solver
    for j=1:J

        %project the differences onto tangent,normal plane for xi
        x_proj = (M'*[T_v2_1(:,i),T_v2_2(:,i),S1(:,i),S2(:,i),S3(:,i)])';

        %form a vandermonde-like matrix
        V = [];
        for d = 1:ell
            for pp = 0:d
                qq = d - pp;
                if pp==0 && qq==0
                    continue
                end
                V = [V, ((x_proj(1,2:end).^pp).*(x_proj(2,2:end).^qq))'];
            end
        end

        %solve for the coefficient
        coeff1 = (V\(x_proj(3,2:end))');
        a1 = coeff1(1:2);
        coeff2 = (V\(x_proj(4,2:end))');
        b1 = coeff2(1:2);
        coeff3 = (V\(x_proj(5,2:end))');
        c1 = coeff3(1:2);

        %update tangent vector
        t_new1 = T_v2_1(:,i) + a1(2)*S1(:,i) + b1(2)*S2(:,i) + c1(2)*S3(:,i);
        t_new2 = T_v2_2(:,i) + a1(1)*S1(:,i) + b1(1)*S2(:,i) + c1(1)*S3(:,i);

        t_new1 = t_new1/norm(t_new1);
        t_new2 = t_new2/norm(t_new2);

        T_v2_1(:,i) = t_new1;
        T_v2_2(:,i) = t_new2;

        %update normal vector
        s_new = null([T_v2_1(:,i),T_v2_2(:,i)]');
        S1(:,i) = s_new(:,1)/norm(s_new(:,1));
        S2(:,i) = s_new(:,2)/norm(s_new(:,2));
        S3(:,i) = s_new(:,3)/norm(s_new(:,3));

    end
end


%% step 8 - estimate derivatives

%solve a linear system for du (t derivatives and x derivatives)
du_t = zeros(2,size(X,1));
du_x = zeros(2,size(X,1));
for pp=1:size(X,1)
    A = [T_v2_1(1:2,pp),T_v2_2(1:2,pp)]';
    b_t = [T_v2_1(4,pp); T_v2_2(4,pp)];
    b_x = [T_v2_1(5,pp); T_v2_2(5,pp)];
    du_t(:,pp) = A\b_t;
    du_x(:,pp) = A\b_x;
end

%ratio of tangent vector to compute x_t and y_t
X = [X, du_t(1,:)', du_t(2,:)', du_x(2,:)'];


%% step 10 - svd to estimate tangents and normals

%keep track of normal directions
S1 = zeros(8,size(X,1));
S2 = zeros(8,size(X,1));
S3 = zeros(8,size(X,1));
S4 = zeros(8,size(X,1));
S5 = zeros(8,size(X,1));
S6 = zeros(8,size(X,1));
T_v3_1 = zeros(8,size(X,1));
T_v3_2 = zeros(8,size(X,1));

%nearest neighbors
[idx_all, ~] = knnsearch(X, X, 'K', K);

%for each data point x_i, we will do the following:
for i=1:size(X,1)

    %find the K nearest neighbors of x_i and store them in neighbor
    index = idx_all(i,:);
    neighbor = X(index,:);

    %construct the matrix M_i
    M = neighbor' - X(i,:)';

    %take the SVD of M
    [U,~,~] = svd(M);

    %track the left singular vector corresponding to largest sv (tangent vector approximation)
    T_v3_1(:,i) = U(:,1);
    T_v3_2(:,i) = U(:,2);

    %do it for the normal vectors
    S_temp = null([T_v3_1(:,i),T_v3_2(:,i)]');
    S1(:,i) = S_temp(:,1)/norm(S_temp(:,1));
    S2(:,i) = S_temp(:,2)/norm(S_temp(:,2));
    S3(:,i) = S_temp(:,3)/norm(S_temp(:,3));
    S4(:,i) = S_temp(:,4)/norm(S_temp(:,4));
    S5(:,i) = S_temp(:,5)/norm(S_temp(:,5));
    S6(:,i) = S_temp(:,6)/norm(S_temp(:,6));


    %% step 11 - run GMLS to improve SVD approximation

    %iterate GMLS solver
    for j=1:J

        %project the differences onto tangent,normal plane for xi
        x_proj = (M'*[T_v3_1(:,i),T_v3_2(:,i),S1(:,i),S2(:,i),S3(:,i),S4(:,i),S5(:,i),S6(:,i)])';

        %form a vandermonde-like matrix
        V = [];
        for d = 1:ell
            for pp = 0:d
                qq = d - pp;
                if pp==0 && qq==0
                    continue
                end
                V = [V, ((x_proj(1,2:end).^pp).*(x_proj(2,2:end).^qq))'];
            end
        end

        %solve for the coefficient
        coeff1 = (V\(x_proj(3,2:end))');
        a1 = coeff1(1:2);
        coeff2 = (V\(x_proj(4,2:end))');
        b1 = coeff2(1:2);
        coeff3 = (V\(x_proj(5,2:end))');
        c1 = coeff3(1:2);
        coeff5 = (V\(x_proj(6,2:end))');
        d1 = coeff5(1:2);
        coeff6 = (V\(x_proj(7,2:end))');
        e1 = coeff6(1:2);
        coeff7 = (V\(x_proj(8,2:end))');
        f1 = coeff7(1:2);

        %update tangent vector
        t_new1 = T_v3_1(:,i) + a1(2)*S1(:,i) + b1(2)*S2(:,i) + c1(2)*S3(:,i) + d1(2)*S4(:,i) + e1(2)*S5(:,i) + f1(2)*S6(:,i);
        t_new2 = T_v3_2(:,i) + a1(1)*S1(:,i) + b1(1)*S2(:,i) + c1(1)*S3(:,i) + d1(1)*S4(:,i) + e1(1)*S5(:,i) + f1(1)*S6(:,i);

        t_new1 = t_new1/norm(t_new1);
        t_new2 = t_new2/norm(t_new2);

        T_v3_1(:,i) = t_new1;
        T_v3_2(:,i) = t_new2;

        %update normal vector
        s_new = null([T_v3_1(:,i),T_v3_2(:,i)]');
        S1(:,i) = s_new(:,1)/norm(s_new(:,1));
        S2(:,i) = s_new(:,2)/norm(s_new(:,2));
        S3(:,i) = s_new(:,3)/norm(s_new(:,3));
        S4(:,i) = s_new(:,4)/norm(s_new(:,4));
        S5(:,i) = s_new(:,5)/norm(s_new(:,5));
        S6(:,i) = s_new(:,6)/norm(s_new(:,6));

    end
end


%% step 12 - estimate prolongated infinitesimal generator

%initialize data matrix P
P = zeros(6*size(X,1),12);

%populate matrix P
for i=1:size(X,1)

    %first normal vector
    P(i,1) = S1(1,i);
    P(i,2) = S1(1,i)*X(i,1) - S1(4,i)*X(i,4) - 2*S1(6,i)*X(i,6) - S1(7,i)*X(i,7);
    P(i,3) = S1(1,i)*X(i,2) - S1(7,i)*X(i,6) - 2*S1(8,i)*X(i,7) - S1(5,i)*X(i,4);
    P(i,4) = S1(1,i)*X(i,3) - S1(4,i)*X(i,4)^2 - S1(7,i)*(2*X(i,4)*X(i,7) + X(i,6)*X(i,5)) - S1(8,i)*(X(i,4)*X(i,8) + 2*X(i,7)*X(i,5)) - 3*S1(6,i)*X(i,4)*X(i,6) - S1(5,i)*X(i,4)*X(i,5);
    P(i,5) = S1(2,i);
    P(i,6) = S1(2,i)*X(i,1) - 2*S1(6,i)*X(i,7) - S1(4,i)*X(i,5) - S1(7,i)*X(i,8);
    P(i,7) = S1(2,i)*X(i,2) - S1(5,i)*X(i,5) - 2*S1(8,i)*X(i,8) - S1(7,i)*X(i,7);
    P(i,8) = S1(2,i)*X(i,3) - S1(5,i)*X(i,5)^2 - S1(6,i)*(2*X(i,4)*X(i,7) + X(i,6)*X(i,5)) - S1(7,i)*(X(i,4)*X(i,8) + 2*X(i,7)*X(i,5)) - S1(4,i)*X(i,4)*X(i,5) - 3*S1(8,i)*X(i,5)*X(i,8);
    P(i,9) = S1(3,i);
    P(i,10) = S1(4,i) + S1(3,i)*X(i,1);
    P(i,11) = S1(5,i) + S1(3,i)*X(i,2);
    P(i,12) = S1(3,i)*X(i,3) + S1(4,i)*X(i,4) + S1(6,i)*X(i,6) + S1(7,i)*X(i,7) + S1(5,i)*X(i,5) + S1(8,i)*X(i,8);

    %second normal vector
    P(1*size(X,1)+i,1) = S2(1,i);
    P(1*size(X,1)+i,2) = S2(1,i)*X(i,1) - S2(4,i)*X(i,4) - 2*S2(6,i)*X(i,6) - S2(7,i)*X(i,7);
    P(1*size(X,1)+i,3) = S2(1,i)*X(i,2) - S2(7,i)*X(i,6) - 2*S2(8,i)*X(i,7) - S2(5,i)*X(i,4);
    P(1*size(X,1)+i,4) = S2(1,i)*X(i,3) - S2(4,i)*X(i,4)^2 - S2(7,i)*(2*X(i,4)*X(i,7) + X(i,6)*X(i,5)) - S2(8,i)*(X(i,4)*X(i,8) + 2*X(i,7)*X(i,5)) - 3*S2(6,i)*X(i,4)*X(i,6) - S2(5,i)*X(i,4)*X(i,5);
    P(1*size(X,1)+i,5) = S2(2,i);
    P(1*size(X,1)+i,6) = S2(2,i)*X(i,1) - 2*S2(6,i)*X(i,7) - S2(4,i)*X(i,5) - S2(7,i)*X(i,8);
    P(1*size(X,1)+i,7) = S2(2,i)*X(i,2) - S2(5,i)*X(i,5) - 2*S2(8,i)*X(i,8) - S2(7,i)*X(i,7);
    P(1*size(X,1)+i,8) = S2(2,i)*X(i,3) - S2(5,i)*X(i,5)^2 - S2(6,i)*(2*X(i,4)*X(i,7) + X(i,6)*X(i,5)) - S2(7,i)*(X(i,4)*X(i,8) + 2*X(i,7)*X(i,5)) - S2(4,i)*X(i,4)*X(i,5) - 3*S2(8,i)*X(i,5)*X(i,8);
    P(1*size(X,1)+i,9) = S2(3,i);
    P(1*size(X,1)+i,10) = S2(4,i) + S2(3,i)*X(i,1);
    P(1*size(X,1)+i,11) = S2(5,i) + S2(3,i)*X(i,2);
    P(1*size(X,1)+i,12) = S2(3,i)*X(i,3) + S2(4,i)*X(i,4) + S2(6,i)*X(i,6) + S2(7,i)*X(i,7) + S2(5,i)*X(i,5) + S2(8,i)*X(i,8);

    %third normal vector
    P(2*size(X,1)+i,1) = S3(1,i);
    P(2*size(X,1)+i,2) = S3(1,i)*X(i,1) - S3(4,i)*X(i,4) - 2*S3(6,i)*X(i,6) - S3(7,i)*X(i,7);
    P(2*size(X,1)+i,3) = S3(1,i)*X(i,2) - S3(7,i)*X(i,6) - 2*S3(8,i)*X(i,7) - S3(5,i)*X(i,4);
    P(2*size(X,1)+i,4) = S3(1,i)*X(i,3) - S3(4,i)*X(i,4)^2 - S3(7,i)*(2*X(i,4)*X(i,7) + X(i,6)*X(i,5)) - S3(8,i)*(X(i,4)*X(i,8) + 2*X(i,7)*X(i,5)) - 3*S3(6,i)*X(i,4)*X(i,6) - S3(5,i)*X(i,4)*X(i,5);
    P(2*size(X,1)+i,5) = S3(2,i);
    P(2*size(X,1)+i,6) = S3(2,i)*X(i,1) - 2*S3(6,i)*X(i,7) - S3(4,i)*X(i,5) - S3(7,i)*X(i,8);
    P(2*size(X,1)+i,7) = S3(2,i)*X(i,2) - S3(5,i)*X(i,5) - 2*S3(8,i)*X(i,8) - S3(7,i)*X(i,7);
    P(2*size(X,1)+i,8) = S3(2,i)*X(i,3) - S3(5,i)*X(i,5)^2 - S3(6,i)*(2*X(i,4)*X(i,7) + X(i,6)*X(i,5)) - S3(7,i)*(X(i,4)*X(i,8) + 2*X(i,7)*X(i,5)) - S3(4,i)*X(i,4)*X(i,5) - 3*S3(8,i)*X(i,5)*X(i,8);
    P(2*size(X,1)+i,9) = S3(3,i);
    P(2*size(X,1)+i,10) = S3(4,i) + S3(3,i)*X(i,1);
    P(2*size(X,1)+i,11) = S3(5,i) + S3(3,i)*X(i,2);
    P(2*size(X,1)+i,12) = S3(3,i)*X(i,3) + S3(4,i)*X(i,4) + S3(6,i)*X(i,6) + S3(7,i)*X(i,7) + S3(5,i)*X(i,5) + S3(8,i)*X(i,8);

    %fourth normal vector
    P(3*size(X,1)+i,1) = S4(1,i);
    P(3*size(X,1)+i,2) = S4(1,i)*X(i,1) - S4(4,i)*X(i,4) - 2*S4(6,i)*X(i,6) - S4(7,i)*X(i,7);
    P(3*size(X,1)+i,3) = S4(1,i)*X(i,2) - S4(7,i)*X(i,6) - 2*S4(8,i)*X(i,7) - S4(5,i)*X(i,4);
    P(3*size(X,1)+i,4) = S4(1,i)*X(i,3) - S4(4,i)*X(i,4)^2 - S4(7,i)*(2*X(i,4)*X(i,7) + X(i,6)*X(i,5)) - S4(8,i)*(X(i,4)*X(i,8) + 2*X(i,7)*X(i,5)) - 3*S4(6,i)*X(i,4)*X(i,6) - S4(5,i)*X(i,4)*X(i,5);
    P(3*size(X,1)+i,5) = S4(2,i);
    P(3*size(X,1)+i,6) = S4(2,i)*X(i,1) - 2*S4(6,i)*X(i,7) - S4(4,i)*X(i,5) - S4(7,i)*X(i,8);
    P(3*size(X,1)+i,7) = S4(2,i)*X(i,2) - S4(5,i)*X(i,5) - 2*S4(8,i)*X(i,8) - S4(7,i)*X(i,7);
    P(3*size(X,1)+i,8) = S4(2,i)*X(i,3) - S4(5,i)*X(i,5)^2 - S4(6,i)*(2*X(i,4)*X(i,7) + X(i,6)*X(i,5)) - S4(7,i)*(X(i,4)*X(i,8) + 2*X(i,7)*X(i,5)) - S4(4,i)*X(i,4)*X(i,5) - 3*S4(8,i)*X(i,5)*X(i,8);
    P(3*size(X,1)+i,9) = S4(3,i);
    P(3*size(X,1)+i,10) = S4(4,i) + S4(3,i)*X(i,1);
    P(3*size(X,1)+i,11) = S4(5,i) + S4(3,i)*X(i,2);
    P(3*size(X,1)+i,12) = S4(3,i)*X(i,3) + S4(4,i)*X(i,4) + S4(6,i)*X(i,6) + S4(7,i)*X(i,7) + S4(5,i)*X(i,5) + S4(8,i)*X(i,8);

    %fifth normal vector
    P(4*size(X,1)+i,1) = S5(1,i);
    P(4*size(X,1)+i,2) = S5(1,i)*X(i,1) - S5(4,i)*X(i,4) - 2*S5(6,i)*X(i,6) - S5(7,i)*X(i,7);
    P(4*size(X,1)+i,3) = S5(1,i)*X(i,2) - S5(7,i)*X(i,6) - 2*S5(8,i)*X(i,7) - S5(5,i)*X(i,4);
    P(4*size(X,1)+i,4) = S5(1,i)*X(i,3) - S5(4,i)*X(i,4)^2 - S5(7,i)*(2*X(i,4)*X(i,7) + X(i,6)*X(i,5)) - S5(8,i)*(X(i,4)*X(i,8) + 2*X(i,7)*X(i,5)) - 3*S5(6,i)*X(i,4)*X(i,6) - S5(5,i)*X(i,4)*X(i,5);
    P(4*size(X,1)+i,5) = S5(2,i);
    P(4*size(X,1)+i,6) = S5(2,i)*X(i,1) - 2*S5(6,i)*X(i,7) - S5(4,i)*X(i,5) - S5(7,i)*X(i,8);
    P(4*size(X,1)+i,7) = S5(2,i)*X(i,2) - S5(5,i)*X(i,5) - 2*S5(8,i)*X(i,8) - S5(7,i)*X(i,7);
    P(4*size(X,1)+i,8) = S5(2,i)*X(i,3) - S5(5,i)*X(i,5)^2 - S5(6,i)*(2*X(i,4)*X(i,7) + X(i,6)*X(i,5)) - S5(7,i)*(X(i,4)*X(i,8) + 2*X(i,7)*X(i,5)) - S5(4,i)*X(i,4)*X(i,5) - 3*S5(8,i)*X(i,5)*X(i,8);
    P(4*size(X,1)+i,9) = S5(3,i);
    P(4*size(X,1)+i,10) = S5(4,i) + S5(3,i)*X(i,1);
    P(4*size(X,1)+i,11) = S5(5,i) + S5(3,i)*X(i,2);
    P(4*size(X,1)+i,12) = S5(3,i)*X(i,3) + S5(4,i)*X(i,4) + S5(6,i)*X(i,6) + S5(7,i)*X(i,7) + S5(5,i)*X(i,5) + S5(8,i)*X(i,8);

    %sixth normal vector
    P(5*size(X,1)+i,1) = S6(1,i);
    P(5*size(X,1)+i,2) = S6(1,i)*X(i,1) - S6(4,i)*X(i,4) - 2*S6(6,i)*X(i,6) - S6(7,i)*X(i,7);
    P(5*size(X,1)+i,3) = S6(1,i)*X(i,2) - S6(7,i)*X(i,6) - 2*S6(8,i)*X(i,7) - S6(5,i)*X(i,4);
    P(5*size(X,1)+i,4) = S6(1,i)*X(i,3) - S6(4,i)*X(i,4)^2 - S6(7,i)*(2*X(i,4)*X(i,7) + X(i,6)*X(i,5)) - S6(8,i)*(X(i,4)*X(i,8) + 2*X(i,7)*X(i,5)) - 3*S6(6,i)*X(i,4)*X(i,6) - S6(5,i)*X(i,4)*X(i,5);
    P(5*size(X,1)+i,5) = S6(2,i);
    P(5*size(X,1)+i,6) = S6(2,i)*X(i,1) - 2*S6(6,i)*X(i,7) - S6(4,i)*X(i,5) - S6(7,i)*X(i,8);
    P(5*size(X,1)+i,7) = S6(2,i)*X(i,2) - S6(5,i)*X(i,5) - 2*S6(8,i)*X(i,8) - S6(7,i)*X(i,7);
    P(5*size(X,1)+i,8) = S6(2,i)*X(i,3) - S6(5,i)*X(i,5)^2 - S6(6,i)*(2*X(i,4)*X(i,7) + X(i,6)*X(i,5)) - S6(7,i)*(X(i,4)*X(i,8) + 2*X(i,7)*X(i,5)) - S6(4,i)*X(i,4)*X(i,5) - 3*S6(8,i)*X(i,5)*X(i,8);
    P(5*size(X,1)+i,9) = S6(3,i);
    P(5*size(X,1)+i,10) = S6(4,i) + S6(3,i)*X(i,1);
    P(5*size(X,1)+i,11) = S6(5,i) + S6(3,i)*X(i,2);
    P(5*size(X,1)+i,12) = S6(3,i)*X(i,3) + S6(4,i)*X(i,4) + S6(6,i)*X(i,6) + S6(7,i)*X(i,7) + S6(5,i)*X(i,5) + S6(8,i)*X(i,8);
end

%take the svd to approximate nullspace
[~,sv,V] = svd(P,'econ');

%singular values
sv = diag(sv)

%approximation of the coefficients
c= V(:,end);


%% nice visualization

%stuff
c1 = 2*c/c(2)

%semilog plot of singular values
figure(8)
semilogy(sv,'k.-','LineWidth',1.5,'markersize',40)
xlabel('Index')
ylabel('Singular Value')
grid on
set(gca,'fontsize',15)

%prepare labels for the bar plots
nCoeffs = length(c1);
labels = arrayfun(@(k) sprintf('c%d', k-1), 1:nCoeffs, 'UniformOutput', false);

%bar plot for c1
figure(9)
bar(c1, 'FaceColor', [1 0.5 0])
xticks(1:nCoeffs)
xticklabels(labels)
xtickangle(45)
grid on
set(gca,'fontsize',15)
ylim([-3 3])
ylabel('Value')




