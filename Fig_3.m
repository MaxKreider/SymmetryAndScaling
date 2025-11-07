%% setup

%mr clean, gravy why you flow so mean
clc
clf
clear

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
a = -2;
b = 1;


%% step 1 - generate data

%generate data
if data_type == 2

    %random values and normalize
    x = rand(N,1)*(b-a) + a;
else

    %fixed equally spaced values
    x = a:(b-a)/N:b-b/N;
end

%sort the data
[~,index] = sort(x);
x = x(index);

%use knowledge of true solution
X(:,1) = x';
X(:,2) = exp(x');


%% step 2 - estimate tangent and normal vectors via SVD (rough approximation)

%keep track of normal directions
S = zeros(2,size(X(:,1),1));
T = zeros(2,size(X(:,1),1));

%nearest neighbors
[idx_all, ~] = knnsearch(X, X, 'K', K);

%for each data point x_i, we will do the following:
for i=1:size(X(:,1),1)

    %find the K nearest neighbors of x_i and store them in neighbor
    index = idx_all(i,:);
    neighbor = X(index,:);

    %construct the matrix M_i
    M = neighbor' - X(i,:)';

    %take the SVD of M
    [U,~,~] = svd(M);

    %track the left singular vector corresponding to largest sv (tangent vector approximation)
    T(:,i) = U(:,1);

    %normalize it
    T(:,i) = T(:,i)/norm(T(:,i));

    %assign a normal vector and normalize it
    S(:,i) = [-T(2,i); T(1,i)];
    S(:,i) = S(:,i)/norm(S(:,i));


    %% step 3 - implement GMLS to improve the SVD approximation


    %iterate GMLS solver
    for j=1:J

        %project the differences onto tangent,normal plane for xi
        x_proj = (M'*[T(:,i),S(:,i)])';

        %form a vandermonde-like matrix
        V = repmat(x_proj(1,2:end)',[1 ell]);
        for p=1:size(V,2)
            V(:,p) = V(:,p).^p;
        end

        %solve for the coefficients
        coeff = (V\(x_proj(2,2:end))');
        a1 = coeff(1);

        %update tangent vector
        t_new = T(:,i) + a1*S(:,i);
        t_new = t_new/norm(t_new);
        T(:,i) = t_new;

        %update normal vector
        s_new = [-T(2,i); T(1,i)];
        s_new = s_new/norm(s_new);
        S(:,i) = s_new;

    end
end


%% step 4 - compute the derivatives

%ratio of tangent vector to compute u_x
X = [X, (T(2,:)./T(1,:))'];


%% step 5 - estimate tangent and normal vectors of jet space via SVD (rough approximation)

%keep track of normal directions
S1 = zeros(3,size(X(:,1),1));
S2 = zeros(3,size(X(:,1),1));
T_v2 = zeros(3,size(X(:,1),1));

%nearest neighbors
[idx_all, ~] = knnsearch(X, X, 'K', K);

%for each data point x_i, we will do the following:
for i=1:size(X(:,1),1)

    %find the K nearest neighbors of x_i and store them in neighbor
    index = idx_all(i,:);
    neighbor = X(index,:);

    %construct the matrix M_i
    M = neighbor' - X(i,:)';

    %take the SVD of M
    [U,~,~] = svd(M);

    %track the left singular vector corresponding to largest sv (tangent vector approximation)
    T_v2(:,i) = U(:,1);

    %normalize it
    T_v2(:,i) = T_v2(:,i)/norm(T_v2(:,i));

    %assign a normal vector and normalize it
    S_temp = null(T_v2(:,i)');
    S1(:,i) = S_temp(:,1)/norm(S_temp(:,1));
    S2(:,i) = S_temp(:,2)/norm(S_temp(:,2));


    %% step 6 - implement GMLS to improve the SVD approximation in the jet space


    %iterate GMLS solver
    for j=1:J

        %project the differences onto tangent,normal plane for xi
        x_proj = (M'*[T_v2(:,i),S1(:,i),S2(:,i)])';

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

        %update tangent vector
        t_new = T_v2(:,i) + a1_S1*S1(:,i) + a1_S2*S2(:,i);
        t_new = t_new/norm(t_new);
        T_v2(:,i) = t_new;

        %update normal vector
        s_new = null(T_v2(:,i)');
        S1(:,i) = s_new(:,1)/norm(s_new(:,1));
        S2(:,i) = s_new(:,2)/norm(s_new(:,2));

    end
end


%% step 7 - estimate prolongated infinitesimal generator

%initialize data matrix P
P = zeros(2*size(X(:,1),1),6);

%populate matrix P
for i=1:size(X(:,1),1)

    %first normal vector
    P(i,1) = S1(1,i)*1;
    P(i,2) = S1(1,i)*X(i,1) - S1(3,i)*X(i,3);
    P(i,3) = S1(1,i)*X(i,2) - S1(3,i)*X(i,3)^2;
    P(i,4) = S1(2,i)*1;
    P(i,5) = S1(2,i)*X(i,1) + S1(3,i);
    P(i,6) = S1(2,i)*X(i,2) + S1(3,i)*X(i,3);

    %second normal vector
    P(size(X(:,1),1)+i,1) = S2(1,i)*1;
    P(size(X(:,1),1)+i,2) = S2(1,i)*X(i,1) - S2(3,i)*X(i,3);
    P(size(X(:,1),1)+i,3) = S2(1,i)*X(i,2) - S2(3,i)*X(i,3)^2;
    P(size(X(:,1),1)+i,4) = S2(2,i)*1;
    P(size(X(:,1),1)+i,5) = S2(2,i)*X(i,1) + S2(3,i);
    P(size(X(:,1),1)+i,6) = S2(2,i)*X(i,2) + S2(3,i)*X(i,3);
end


%take the svd to approximate nullspace
[~,ss,V] = svd(P,'econ');

%singular values
sv = diag(ss);

%approximation of the coefficients
c= V(:,end);


%% nice visualization

%stuff
c1 = c/c(1);

% 1. Semilog plot of singular values
figure(1)
semilogy(sv,'k.-','LineWidth',1.5,'markersize',40)
xlabel('Index')
ylabel('Singular Value')
grid on
set(gca,'fontsize',15)

% Prepare labels for the bar plots
nCoeffs = length(c1);
labels = arrayfun(@(k) sprintf('c%d', k-1), 1:nCoeffs, 'UniformOutput', false);

% 2. Bar plot for c1
figure(2)
bar(c1, 'FaceColor', [1 0.5 0])
xticks(1:nCoeffs)
xticklabels(labels)
xtickangle(45)
grid on
set(gca,'fontsize',15)
ylim([-0.2 1.2])
ylabel('Value')

