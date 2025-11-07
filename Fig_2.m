%% setup

%mr clean
clc
clear

%choose data type (1 = fixed equal spacing, 2 = random)
data_type = 2;

%number of data points to consider
N = 500;

%number of nearest neighbors to consider
K = 15;

%degree of local polynomial fit
ell = 4;

%number of GMLS iterations to perform
J = 2;

%domain
a = -1;
b = 1;

%uniformly distributed random samples on [1, 2]
nalpha = numel(1:0.001:2);
alpha = 1 + (2 - 1) * rand(1, nalpha);

%initialize data matrix P
P = zeros(2*N*length(alpha),6);

%initialize big data matrix
X_big = [];

%loop over a family
for count = 1:length(alpha)

    %% step 1 - generate data

    %reset X
    X = zeros(N,2);

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
    X(:,2) = alpha(count)*exp(x');


    %% step 2 - estimate tangent and normal vectors via SVD (rough approximation)

    %keep track of normal directions
    S = zeros(2,N);
    T = zeros(2,N);

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

        %normal vector
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
    X_big = [X_big; [X, (T(2,:)./T(1,:))']];

end


%% step 5 - estimate tangent and normal vectors of jet space via SVD (rough approximation)

%keep track of normal directions
S = zeros(3,N*length(alpha));
T1 = zeros(3,N*length(alpha));
T2 = zeros(3,N*length(alpha));

%nearest neighbors
[idx_all, ~] = knnsearch(X_big, X_big, 'K', K);

%for each data point x_i, we will do the following:
for i=1:N*length(alpha)

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

    %normalize it
    T1(:,i) = T1(:,i)/norm(T1(:,i));
    T2(:,i) = T2(:,i)/norm(T2(:,i));

    %find normal vector
    S_temp = null([T1(:,i),T2(:,i)]');
    S(:,i) = S_temp/norm(S_temp);

end


%% step 6 - implement GMLS to improve the SVD approximation in the jet space

%update nearest neighbors
K = 25;
ell = 3;

%nearest neighbors
[idx_all, ~] = knnsearch(X_big, X_big, 'K', K);

%loop over everyone
for i=1:N*length(alpha)

    %iterate GMLS solver
    for j=1:J

        %find the K nearest neighbors of x_i and store them in neighbor
        index = idx_all(i,:);
        neighbor = X_big(index,:);

        %construct the matrix M_i
        M = neighbor' - X_big(i,:)';

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


%% step 7 - estimate prolongated infinitesimal generator

%populate matrix P
for i=1:N*length(alpha)

    %first normal vector
    P(i,1) = S(1,i)*1;
    P(i,2) = S(1,i)*X_big(i,1) - S(3,i)*X_big(i,3);
    P(i,3) = S(1,i)*X_big(i,2) - S(3,i)*X_big(i,3)^2;
    P(i,4) = S(2,i)*1;
    P(i,5) = S(2,i)*X_big(i,1) + S(3,i);
    P(i,6) = S(2,i)*X_big(i,2) + S(3,i)*X_big(i,3);
end


%take the svd to approximate nullspace
[~,ss,V] = svd(P,'econ');

sv = diag(ss)

%approximation of the coefficients
c= V(:,end);


%% nice visualization

%stuff
c1 = c;
c2 = V(:,end-1);

c1=c1/c1(1)
c2=c2/c2(end)

%semilog plot of singular values
figure(1)
semilogy(sv,'k.-','LineWidth',1.5,'markersize',40)
xlabel('Index')
ylabel('Singular Value')
grid on
set(gca,'fontsize',15)

%prepare labels for the bar plots
nCoeffs = length(c1);
labels = arrayfun(@(k) sprintf('c%d', k-1), 1:nCoeffs, 'UniformOutput', false);

%bar plot for c1
figure(2)
bar(c1, 'FaceColor', [1 0.5 0])
xticks(1:nCoeffs)
xticklabels(labels)
xtickangle(45)
grid on
set(gca,'fontsize',15)
ylim([-1.5 1.5])
ylabel('Value')

%bar plot for c2
figure(3)
bar(c2, 'FaceColor', [1 0.5 0])
xticks(1:nCoeffs)
xticklabels(labels)
xtickangle(45)
grid on
set(gca,'fontsize',15)
ylim([-1.5 1.5])
ylabel('Value')


