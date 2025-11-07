%% setup

%mr clean, gravy why you flow so mean
clc
clf
clear
close all

%choose data type fixed equal spacing
data_type = 2;

%number of data points to consider ^2
N = 70-1;

%number of nearest neighbors to consider
K = 20;

%degree of local polynomial fit
ell = 2;

%number of GMLS iterations to perform
J = 2;

%domain
a = -1/2;
b = 1/2;


%% step 1 - generate data on the circle: u(t,x) = t + x

%generate data
if data_type == 1

    %fixed equally spaced values
    t = a:2/N:b;
    [t,x] = meshgrid(t,t);
    X(:,1) = t(:);
    X(:,2) = x(:);

else

    % uniformly distributed random samples in [a,b]^2
    X = a + (b - a) * rand(N^2, 2);

end

%include u
X = [X, sin(X(:,1)+X(:,2))];


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

        %find the K nearest neighbors of x_i and store them in neighbor
        index = knnsearch(X,X(i,:),'K',K);
        neighbor = X(index(1:end),:);

        %construct the matrix M_i
        M = X(i,:)' - neighbor';

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


%% step 7 - estimate prolongated infinitesimal generator

%initialize data matrix P
P = zeros(3*size(X,1),12);

%populate matrix P
for i=1:size(X,1)

    %first normal vector
    P(i,1) = S1(1,i);
    P(i,2) = S1(1,i)*X(i,1) - S1(4,i)*X(i,4);
    P(i,3) = S1(1,i)*X(i,2) - S1(5,i)*X(i,4);
    P(i,4) = S1(1,i)*X(i,3) - S1(4,i)*X(i,4)^2 - S1(5,i)*X(i,5)*X(i,4);
    P(i,5) = S1(2,i);
    P(i,6) = S1(2,i)*X(i,1) - S1(4,i)*X(i,5);
    P(i,7) = S1(2,i)*X(i,2) - S1(5,i)*X(i,5);
    P(i,8) = S1(2,i)*X(i,3)  - S1(4,i)*X(i,4)*X(i,5) - S1(5,i)*X(i,5)^2;
    P(i,9) = S1(3,i);
    P(i,10) = S1(3,i)*X(i,1) + S1(4,i);
    P(i,11) = S1(3,i)*X(i,2) + S1(5,i);
    P(i,12) = S1(3,i)*X(i,3) + S1(4,i)*X(i,4) + S1(5,i)*X(i,5);

    %second normal vector
    P(size(X,1)+i,1) = S2(1,i);
    P(size(X,1)+i,2) = S2(1,i)*X(i,1) - S2(4,i)*X(i,4);
    P(size(X,1)+i,3) = S2(1,i)*X(i,2) - S2(5,i)*X(i,4);
    P(size(X,1)+i,4) = S2(1,i)*X(i,3) - S2(4,i)*X(i,4)^2 - S2(5,i)*X(i,5)*X(i,4);
    P(size(X,1)+i,5) = S2(2,i);
    P(size(X,1)+i,6) = S2(2,i)*X(i,1) - S2(4,i)*X(i,5);
    P(size(X,1)+i,7) = S2(2,i)*X(i,2) - S2(5,i)*X(i,5);
    P(size(X,1)+i,8) = S2(2,i)*X(i,3)  - S2(4,i)*X(i,4)*X(i,5) - S2(5,i)*X(i,5)^2;
    P(size(X,1)+i,9) = S2(3,i);
    P(size(X,1)+i,10) = S2(3,i)*X(i,1) + S2(4,i);
    P(size(X,1)+i,11) = S2(3,i)*X(i,2) + S2(5,i);
    P(size(X,1)+i,12) = S2(3,i)*X(i,3) + S2(4,i)*X(i,4) + S2(5,i)*X(i,5);

    %third normal vector
    P(2*size(X,1)+i,1) = S3(1,i);
    P(2*size(X,1)+i,2) = S3(1,i)*X(i,1) - S3(4,i)*X(i,4);
    P(2*size(X,1)+i,3) = S3(1,i)*X(i,2) - S3(5,i)*X(i,4);
    P(2*size(X,1)+i,4) = S3(1,i)*X(i,3) - S3(4,i)*X(i,4)^2 - S3(5,i)*X(i,5)*X(i,4);
    P(2*size(X,1)+i,5) = S3(2,i);
    P(2*size(X,1)+i,6) = S3(2,i)*X(i,1) - S3(4,i)*X(i,5);
    P(2*size(X,1)+i,7) = S3(2,i)*X(i,2) - S3(5,i)*X(i,5);
    P(2*size(X,1)+i,8) = S3(2,i)*X(i,3)  - S3(4,i)*X(i,4)*X(i,5) - S3(5,i)*X(i,5)^2;
    P(2*size(X,1)+i,9) = S3(3,i);
    P(2*size(X,1)+i,10) = S3(3,i)*X(i,1) + S3(4,i);
    P(2*size(X,1)+i,11) = S3(3,i)*X(i,2) + S3(5,i);
    P(2*size(X,1)+i,12) = S3(3,i)*X(i,3) + S3(4,i)*X(i,4) + S3(5,i)*X(i,5);
end

%take the svd to approximate nullspace
[~,sv,V] = svd(P,'econ');

%singular values
sv = diag(sv)

%approximation of the coefficients
c= V(:,end)


%% nice visualization

%stuff
c1 = 3*c/c(2);

% 1. Semilog plot of singular values
figure(8)
semilogy(sv,'k.-','LineWidth',1.5,'markersize',40)
xlabel('Index')
ylabel('Singular Value')
grid on
set(gca,'fontsize',15)

% Prepare labels for the bar plots
nCoeffs = length(c1);
labels = arrayfun(@(k) sprintf('c%d', k-1), 1:nCoeffs, 'UniformOutput', false);

% 2. Bar plot for c1
figure(9)
bar(c1, 'FaceColor', [1 0.5 0])
xticks(1:nCoeffs)
xticklabels(labels)
xtickangle(45)
grid on
set(gca,'fontsize',15)
ylim([-6 6])
ylabel('Value')


