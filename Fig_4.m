%% setup

%mr clean, gravy why you flow so mean
clc
clear

%choose data type (1 = fixed equal spacing, 2 = random)
data_type = 2;

%number of data points to consider
N = 300;

%number of nearest neighbors to consider
K = 40;

%degree of local polynomial fit
ell = 4;

%number of GMLS iterations to perform
J = 2;

%domain
a = 0*pi;
b = pi;

% number of samples (same as before)
nr = numel(1:0.25:1.3);
ntheta = numel(0:0.05:2*pi);

% uniformly distributed random samples
r0 = 1 + (1.3 - 1) * rand(1, nr);
theta0 = 0 + (2*pi - 0) * rand(1, ntheta);

%initialize big data matrix
X_big = [];

%loop over a family
for count1 = 1:length(r0)
    for count2 = 1:length(theta0)

        %% step 1 - generate data

        %reset X
        X = zeros(N,3);

        %generate data
        if data_type == 2

            %random values and normalize
            x = rand(N,1)*(b-a) + a;
        else

            %fixed equally spaced values
            x = linspace(a, b, N)';
        end

        %use knowledge of true solution
        X(:,1) = x';
        X(:,2) = cos(-x'+theta0(count2))./sqrt(1-(1-1/(r0(count1)^2)).*exp(-2*x'));
        X(:,3) = sin(-x'+theta0(count2))./sqrt(1-(1-1/(r0(count1)^2)).*exp(-2*x'));


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

            %normalize it
            T(:,i) = T(:,i)/norm(T(:,i));

            %choose a fixed orientation by the "first" tangent vector
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

        %ratio of tangent vector to compute u_x
        X_big = [X_big; [X, (T(2,:)./T(1,:))', (T(3,:)./T(1,:))']];


    end

    count1
end


%% step 5 - estimate tangent and normal vectors of jet space via SVD (rough approximation)


%keep track of normal directions
S1 = zeros(5,N*length(r0)*length(theta0));
S2 = zeros(5,N*length(r0)*length(theta0));
T1 = zeros(5,N*length(r0)*length(theta0));
T2 = zeros(5,N*length(r0)*length(theta0));
T3 = zeros(5,N*length(r0)*length(theta0));

%nearest neighbors
[idx_all, ~] = knnsearch(X_big, X_big, 'K', K);

%for each data point x_i, we will do the following:
for i=1:N*length(r0)*length(theta0)

    %find the K nearest neighbors of x_i and store them in neighbor
    index = idx_all(i,:);
    neighbor = X_big(index,:);

    %construct the matrix M_i
    M = neighbor' - X_big(i,:)';

    %take the SVD of M
    [U,~,~] = svd(M);

    %track the left singular vector corresponding to largest sv (tangent vector approximation)
    T1(:,i) = U(:,1);
    T2(:,i) = U(:,2);
    T3(:,i) = U(:,3);

    %normalize it
    T1(:,i) = T1(:,i)/norm(T1(:,i));
    T2(:,i) = T2(:,i)/norm(T2(:,i));
    T3(:,i) = T3(:,i)/norm(T3(:,i));

    %find normal vector
    S_temp = null([T1(:,i),T2(:,i),T3(:,i)]');
    S1(:,i) = S_temp(:,1)/norm(S_temp(:,1));
    S2(:,i) = S_temp(:,2)/norm(S_temp(:,2));

end


%% step 7 - estimate prolongated infinitesimal generator

%initialize data matrix P
P = zeros(2*N*length(r0)*length(theta0),12);

%populate matrix P
for i=1:N*length(r0)*length(theta0)

    %first normal vector
    P(i,1) = S1(1,i);
    P(i,2) = S1(1,i)*X_big(i,1) - X_big(i,4)*S1(4,i) - X_big(i,5)*S1(5,i);
    P(i,3) = S1(1,i)*X_big(i,2) - S1(4,i)*X_big(i,4)^2 - S1(5,i)*X_big(i,4)*X_big(i,5);
    P(i,4) = S1(1,i)*X_big(i,3) - S1(4,i)*X_big(i,4)*X_big(i,5) - S1(5,i)*X_big(i,5)^2;
    P(i,5) = S1(2,i);
    P(i,6) = S1(2,i)*X_big(i,1) + S1(4,i);
    P(i,7) = S1(2,i)*X_big(i,2) + S1(4,i)*X_big(i,4);
    P(i,8) = S1(2,i)*X_big(i,3) + S1(4,i)*X_big(i,5);
    P(i,9) = S1(3,i);
    P(i,10) = S1(3,i)*X_big(i,1) + S1(5,i);
    P(i,11) = S1(3,i)*X_big(i,2) + S1(5,i)*X_big(i,4);
    P(i,12) = S1(3,i)*X_big(i,3) + S1(5,i)*X_big(i,5);

    %second normal vector
    P(N*length(r0)*length(theta0)+i,1) = S2(1,i);
    P(N*length(r0)*length(theta0)+i,2) = S2(1,i)*X_big(i,1) - X_big(i,4)*S2(4,i) - X_big(i,5)*S2(5,i);
    P(N*length(r0)*length(theta0)+i,3) = S2(1,i)*X_big(i,2) - S2(4,i)*X_big(i,4)^2 - S2(5,i)*X_big(i,4)*X_big(i,5);
    P(N*length(r0)*length(theta0)+i,4) = S2(1,i)*X_big(i,3) - S2(4,i)*X_big(i,4)*X_big(i,5) - S2(5,i)*X_big(i,5)^2;
    P(N*length(r0)*length(theta0)+i,5) = S2(2,i);
    P(N*length(r0)*length(theta0)+i,6) = S2(2,i)*X_big(i,1) + S2(4,i);
    P(N*length(r0)*length(theta0)+i,7) = S2(2,i)*X_big(i,2) + S2(4,i)*X_big(i,4);
    P(N*length(r0)*length(theta0)+i,8) = S2(2,i)*X_big(i,3) + S2(4,i)*X_big(i,5);
    P(N*length(r0)*length(theta0)+i,9) = S2(3,i);
    P(N*length(r0)*length(theta0)+i,10) = S2(3,i)*X_big(i,1) + S2(5,i);
    P(N*length(r0)*length(theta0)+i,11) = S2(3,i)*X_big(i,2) + S2(5,i)*X_big(i,4);
    P(N*length(r0)*length(theta0)+i,12) = S2(3,i)*X_big(i,3) + S2(5,i)*X_big(i,5);
end


%take the svd to approximate nullspace
[~,ss,V] = svd(P,'econ');

sv = diag(ss)

%approximation of the coefficients
c= V(:,end);


%% nice visualization

%stuff
c1 = c
c2 = V(:,end-1)

%normalize
c1 = c1/c1(end-1);
c2 = c2/c2(1);

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
ylim([-1.2 1.2])
ylabel('Value')

% 3. Bar plot for c2
figure(3)
bar(c2, 'FaceColor', [1 0.5 0])
xticks(1:nCoeffs)
xticklabels(labels)
xtickangle(45)
grid on
set(gca,'fontsize',15)
ylim([-.2 1.2])
ylabel('Value')


%% error analysis

%true nullspace vector
truth1 = [1 0 0 0 0 0 0 0 0 0 0 0]';
truth2 = [0 0 0 0 0 0 0 -1 0 0 1 0]';
truth = [truth1,truth2];

%approximate
c = [c1,c2];

%error
abs(sin(subspace(truth,c)))