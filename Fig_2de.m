%% setup

%verified

%mr clean
clc
clf
clear

%choose data type (1 = fixed equal spacing, 2 = random)
data_type = 2;

%number of data points to consider
NN = 1000;

%number of trials over which to average
ntrials = 1;

%number of nearest neighbors to consider
K = 10;

%degree of local polynomial fit
ell = 3;

%number of GMLS iterations to perform
J = 2;

%domain
a = -2;
b = 1;

%true nullspace vector
truth = [1 0 0 0 0 1]';

%error vector
err_vec = zeros(ntrials,length(NN));

%loop over multiple trials for each value of N
for aaa = 1:ntrials

    %initialize holder variable
    local_err = zeros(1, length(NN));

    %loop over the number of different samples
    for qqq = 1:length(NN)

        %set number of samples
        N = NN(qqq);


        %% step 1 - generate data

        %generate data
        if data_type == 2

            %random values and normalize
            x = rand(N,1)*(1-(-2)) + (-2);
        else

            %fixed equally spaced values
            x = (-2):(1-(-2))/N:1-1/N;
        end

        %sort the data
        [~,index] = sort(x);
        x = x(index);

        %use knowledge of true solution
        X = zeros(N, 2);
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

        %P'*P
        A = zeros(6,6); 
        for i = 1:N

            % build small local rows for each S vector
            p1 = local_row(X(i,:), S1(:,i));  
            p2 = local_row(X(i,:), S2(:,i));

            % accumulate (P'*P)
            A = A + p1'*p1 + p2'*p2;
        end

        %take the svd to approximate nullspace
        [~,ss,V] = svd(A,'econ');

        %singular values
        sv = diag(ss);

        %approximation of the coefficients
        c= V(:,end);


        %% error analysis

        %compute the error
        local_err(qqq) = abs(sin(subspace(truth,c)));

    end

    %concatenate
    err_vec(aaa, :) = local_err;

end

%do the average
err_vec_avg = mean(err_vec);


%% nice visualization

%normalize
c = c/c(1);

%semilog plot of singular values
figure(1)
semilogy(sv,'k.-','LineWidth',1.5,'markersize',40)
xlabel('Index')
ylabel('Singular Value')
grid on
set(gca,'fontsize',15)

%prepare labels for the bar plots
nCoeffs = length(c);
labels = arrayfun(@(k) sprintf('c%d', k-1), 1:nCoeffs, 'UniformOutput', false);

%bar plot for c
figure(2)
bar(c, 'FaceColor', [1 0.5 0])
xticks(1:nCoeffs)
xticklabels(labels)
xtickangle(45)
grid on
set(gca,'fontsize',15)
ylim([-0.2 1.2])
ylabel('Value')


%% auxiliary functions

function p = local_row(X, S)

    p = zeros(1,6);
    p(1)  = S(1);
    p(2)  = S(1)*X(1) - X(3)*S(3);
    p(3)  = S(1)*X(2) - S(3)*X(3)^2;
    p(4)  = S(2);
    p(5)  = S(2)*X(1) + S(3);
    p(6)  = S(2)*X(2) + S(3)*X(3);
end



