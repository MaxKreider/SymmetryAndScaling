%% setup

%mr clean
clc
clf
clear
close all

%choose data type (1 = fixed equal spacing, 2 = random)
data_type = 2;

%number of data points to consider
N = 1000;

%number of nearest neighbors to consider
K = 10;

%degree of local polynomial fit
ell = 3;

%number of GMLS iterations to perform
J = 2;

%domain
a = 0;
b = 2*pi;


%% step 1 - generate data on the circle: x^2 + y^2 - 1 = 0

%generate data
if data_type == 2

    %random values and normalize
    X = randn(N,2);
    X = X./vecnorm(X,2,2);
else

    %fixed equally spaced values
    x = linspace(a,b,N);
    X(:,1) = cos(x');
    X(:,2) = sin(x');
end

%sort the data, e.g., by parametrizing the circle with an angle theta
theta = angle(X(:,1)+sqrt(-1)*X(:,2))+pi;
[~,index] = sort(theta);
X = X(index,:);
theta = theta(index);

%include time in the data
X = [theta, X];


%% step 2 - estimate tangent and normal vectors via SVD (rough approximation)

%keep track of normal directions
S1 = zeros(3,N);
S2 = zeros(3,N);
T = zeros(3,N);

%nearest neighbors
[idx_all, ~] = knnsearch(X, X, 'K', K);

%for each data point x_i, we will do the following:
for i=1:N

    %find the K nearest neighbors of x_i and store them in neighbor
    index = idx_all(i,:);
    neighbor = X(index,:);

    %construct the matrix M_i
    M = neighbor' - X(i,:)';

    %take the SVD of M
    [U,~,~] = svd(M);

    %track the left singular vector corresponding to largest sv (tangent vector approximation)
    T(:,i) = U(:,1);

    %assign a normal vector and normalize it
    S_temp = null(T(:,i)');
    S1(:,i) = S_temp(:,1)/norm(S_temp(:,1));
    S2(:,i) = S_temp(:,2)/norm(S_temp(:,2));


    %% step 3 - implement GMLS to improve the SVD approximation


    %iterate GMLS solver
    for j=1:J

        %project the differences onto tangent,normal plane for xi
        x_proj = (M'*[T(:,i),S1(:,i),S2(:,i)])';

        %form a vandermonde-like matrix
        V = repmat(x_proj(1,2:end)',[1 ell]);
        for p=1:size(V,2)
            V(:,p) = V(:,p).^p;
        end

        %solve for the coefficients
        coeff1 = (V\(x_proj(2,2:end))');
        a1_S1 = coeff1(1);
        coeff2 = (V\(x_proj(3,2:end))');
        a1_S2 = coeff2(1);

        %update tangent vector
        t_new = T(:,i) + a1_S1*S1(:,i) + a1_S2*S2(:,i);
        t_new = t_new/norm(t_new);
        T(:,i) = t_new;

        %update normal vector
        s_new = null(T(:,i)');
        S1(:,i) = s_new(:,1)/norm(s_new(:,1));
        S2(:,i) = s_new(:,2)/norm(s_new(:,2));

    end
end


%% step 4 - compute the derivatives

%ratio of tangent vector to compute x_t and y_t
X = [X, (T(2,:)'./T(1,:)'), T(3,:)'./T(1,:)'];


%% step 5 - estimate tangent and normal vectors of jet space via SVD (rough approximation)

%keep track of normal directions
S1 = zeros(5,N);
S2 = zeros(5,N);
S3 = zeros(5,N);
S4 = zeros(5,N);
T_v2 = zeros(5,N);

%nearest neighbors
[idx_all, ~] = knnsearch(X, X, 'K', K);

%for each data point x_i, we will do the following:
for i=1:N

    %find the K nearest neighbors of x_i and store them in neighbor
    index = idx_all(i,:);
    neighbor = X(index,:);

    %construct the matrix M_i
    M = neighbor' - X(i,:)';

    %take the SVD of M
    [U,~,~] = svd(M);

    %track the left singular vector corresponding to largest sv (tangent vector approximation)
    T_v2(:,i) = U(:,1);

    %assign a normal vector and normalize it
    S_temp = null(T_v2(:,i)');
    S1(:,i) = S_temp(:,1)/norm(S_temp(:,1));
    S2(:,i) = S_temp(:,2)/norm(S_temp(:,2));
    S3(:,i) = S_temp(:,3)/norm(S_temp(:,3));
    S4(:,i) = S_temp(:,4)/norm(S_temp(:,4));


    %% step 6 - implement GMLS to improve the SVD approximation in the jet space


    %iterate GMLS solver
    for j=1:J

        %project the differences onto tangent,normal plane for xi
        x_proj = (M'*[T_v2(:,i),S1(:,i),S2(:,i),S3(:,i),S4(:,i)])';

        %form a vandermonde-like matrix
        V = repmat(x_proj(1,2:end)',[1 ell]);
        for p=1:size(V,2)
            V(:,p) = V(:,p).^p;
        end

        %solve for the coefficients
        coeff_S1 = (V\(x_proj(2,2:end))');
        a1_S1 = coeff_S1(1);
        coeff_S2 = (V\(x_proj(3,2:end))');
        a1_S2 = coeff_S2(1);
        coeff_S3 = (V\(x_proj(4,2:end))');
        a1_S3 = coeff_S3(1);
        coeff_S4 = (V\(x_proj(5,2:end))');
        a1_S4 = coeff_S4(1);

        %update tangent vector
        t_new = T_v2(:,i) + a1_S1*S1(:,i) + a1_S2*S2(:,i) + a1_S3*S3(:,i) + a1_S4*S4(:,i);
        t_new = t_new/norm(t_new);
        T_v2(:,i) = t_new;

        %update normal vector
        s_new = null(T_v2(:,i)');
        S1(:,i) = s_new(:,1)/norm(s_new(:,1));
        S2(:,i) = s_new(:,2)/norm(s_new(:,2));
        S3(:,i) = s_new(:,3)/norm(s_new(:,3));
        S4(:,i) = s_new(:,4)/norm(s_new(:,4));

    end
end


%% step 7 - compute the derivatives

%ratio of tangent vector to compute x_t and y_t
X = [X, (T_v2(4,:)'./T_v2(1,:)'), T_v2(5,:)'./T_v2(1,:)'];


%% step 8 - estimate prolongated infinitesimal generator

%initialize data matrix P
P = zeros(4*N,12);

%initialize orthogonality check temp
temp = zeros(1,N);

%populate matrix P
for i=1:N

    %first normal vector
    P(i,1) = S1(1,i);
    P(i,2) = S1(1,i)*X(i,1) - X(i,4)*S1(4,i) - X(i,5)*S1(5,i);
    P(i,3) = S1(1,i)*X(i,2) - S1(4,i)*X(i,4)^2 - S1(5,i)*X(i,4)*X(i,5);
    P(i,4) = S1(1,i)*X(i,3) - S1(4,i)*X(i,4)*X(i,5) - S1(5,i)*X(i,5)^2;
    P(i,5) = S1(2,i);
    P(i,6) = S1(2,i)*X(i,1) + S1(4,i);
    P(i,7) = S1(2,i)*X(i,2) + S1(4,i)*X(i,4);
    P(i,8) = S1(2,i)*X(i,3) + S1(4,i)*X(i,5);
    P(i,9) = S1(3,i);
    P(i,10) = S1(3,i)*X(i,1) + S1(5,i);
    P(i,11) = S1(3,i)*X(i,2) + S1(5,i)*X(i,4);
    P(i,12) = S1(3,i)*X(i,3) + S1(5,i)*X(i,5);

    %second normal vector
    P(N+i,1) = S2(1,i);
    P(N+i,2) = S2(1,i)*X(i,1) - X(i,4)*S2(4,i) - X(i,5)*S2(5,i);
    P(N+i,3) = S2(1,i)*X(i,2) - S2(4,i)*X(i,4)^2 - S2(5,i)*X(i,4)*X(i,5);
    P(N+i,4) = S2(1,i)*X(i,3) - S2(4,i)*X(i,4)*X(i,5) - S2(5,i)*X(i,5)^2;
    P(N+i,5) = S2(2,i);
    P(N+i,6) = S2(2,i)*X(i,1) + S2(4,i);
    P(N+i,7) = S2(2,i)*X(i,2) + S2(4,i)*X(i,4);
    P(N+i,8) = S2(2,i)*X(i,3) + S2(4,i)*X(i,5);
    P(N+i,9) = S2(3,i);
    P(N+i,10) = S2(3,i)*X(i,1) + S2(5,i);
    P(N+i,11) = S2(3,i)*X(i,2) + S2(5,i)*X(i,4);
    P(N+i,12) = S2(3,i)*X(i,3) + S2(5,i)*X(i,5);

    %third normal vector
    P(2*N+i,1) = S3(1,i);
    P(2*N+i,2) = S3(1,i)*X(i,1) - X(i,4)*S3(4,i) - X(i,5)*S3(5,i);
    P(2*N+i,3) = S3(1,i)*X(i,2) - S3(4,i)*X(i,4)^2 - S3(5,i)*X(i,4)*X(i,5);
    P(2*N+i,4) = S3(1,i)*X(i,3) - S3(4,i)*X(i,4)*X(i,5) - S3(5,i)*X(i,5)^2;
    P(2*N+i,5) = S3(2,i);
    P(2*N+i,6) = S3(2,i)*X(i,1) + S3(4,i);
    P(2*N+i,7) = S3(2,i)*X(i,2) + S3(4,i)*X(i,4);
    P(2*N+i,8) = S3(2,i)*X(i,3) + S3(4,i)*X(i,5);
    P(2*N+i,9) = S3(3,i);
    P(2*N+i,10) = S3(3,i)*X(i,1) + S3(5,i);
    P(2*N+i,11) = S3(3,i)*X(i,2) + S3(5,i)*X(i,4);
    P(2*N+i,12) = S3(3,i)*X(i,3) + S3(5,i)*X(i,5);

    %fourth normal vector
    P(3*N+i,1) = S4(1,i);
    P(3*N+i,2) = S4(1,i)*X(i,1) - X(i,4)*S4(4,i) - X(i,5)*S4(5,i);
    P(3*N+i,3) = S4(1,i)*X(i,2) - S4(4,i)*X(i,4)^2 - S4(5,i)*X(i,4)*X(i,5);
    P(3*N+i,4) = S4(1,i)*X(i,3) - S4(4,i)*X(i,4)*X(i,5) - S4(5,i)*X(i,5)^2;
    P(3*N+i,5) = S4(2,i);
    P(3*N+i,6) = S4(2,i)*X(i,1) + S4(4,i);
    P(3*N+i,7) = S4(2,i)*X(i,2) + S4(4,i)*X(i,4);
    P(3*N+i,8) = S4(2,i)*X(i,3) + S4(4,i)*X(i,5);
    P(3*N+i,9) = S4(3,i);
    P(3*N+i,10) = S4(3,i)*X(i,1) + S4(5,i);
    P(3*N+i,11) = S4(3,i)*X(i,2) + S4(5,i)*X(i,4);
    P(3*N+i,12) = S4(3,i)*X(i,3) + S4(5,i)*X(i,5);
end

%take the svd to approximate nullspace
[~,sv,V] = svd(P);

%singular values
sv = diag(sv)

%approximation of the coefficients
c= V(:,end)


%% nice visualization

%stuff
c1 = c/c(1);

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
ylim([-1.2 1.2])
ylabel('Value')
