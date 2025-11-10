%% setup

%verified

%mr clean
clc
clf
clear
close all

%choose data type (1 = fixed equal spacing, 2 = random)
data_type = 2;

%number of data points to consider
NN = [80 160 320 640 1280 2560 5120 10240 20480 40960 81920];

%number of trials over which to average
ntrials = 100;

%number of nearest neighbors to consider
K = 10;

%degree of local polynomial fit
ell = 3;

%number of GMLS iterations to perform
J = 2;

%domain
a = 0;
b = 2*pi;

%true nullspace vector
truth = [1 0 0 0 0 0 0 -1 0 0 1 0]';

%error vector
err_vec = zeros(ntrials,length(NN));

%loop over multiple trials for each value of N
parfor aaa = 1:ntrials

    %initialize holder variable
    local_err = zeros(1, length(NN));

    %loop over the number of different samples
    for qqq = 1:length(NN)

        %set number of samples
        N = NN(qqq);

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

            % X(:,1) = exp(x')+x'.*exp(x');
            % X(:,2) = exp(x');
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

        %P'*P
        A = zeros(12,12); 
        for i = 1:N

            % build small local rows for each S vector
            p1 = local_row(X(i,:), S1(:,i));  
            p2 = local_row(X(i,:), S2(:,i));
            p3 = local_row(X(i,:), S3(:,i));
            p4 = local_row(X(i,:), S4(:,i));

            % accumulate (P'*P)
            A = A + p1'*p1 + p2'*p2 + p3'*p3 + p4'*p4;
        end

        %take the svd to approximate nullspace
        [~,sv,V] = svd(A);

        %singular values
        sv = sqrt(diag(sv));

        %approximation of the coefficients
        c= V(:,end);
        c1 = c/c(end-1);


        %% error analysis

        %compute the error
        local_err(qqq) = abs(sin(subspace(truth,c1)));

    end

    %concatenate
    err_vec(aaa, :) = local_err;

end

%do the average
err_vec_avg = mean(err_vec);


%% plot error

%well
plot_N = NN;

%theory
theory = (log(plot_N).^(ell/1).*plot_N.^(1-ell/1));

%visualize the error
figure(1)
loglog(NN,err_vec_avg,'k-','linewidth',3)
hold on
loglog(NN,err_vec_avg,'k.','markersize',40)
loglog(NN,theory*err_vec_avg(1)/theory(1),'r--','linewidth',3)
xlabel('N')
ylabel('||sin(\Theta)||_2')
set(gca,'fontsize',15)
box on
axis square
grid on
legend('Simulation','','Theory')


%% auxiliary functions

function p = local_row(X, S)

    p = zeros(1,12);
    p(1)  = S(1);
    p(2)  = S(1)*X(1) - X(4)*S(4) - X(5)*S(5);
    p(3)  = S(1)*X(2) - S(4)*X(4)^2 - S(5)*X(4)*X(5);
    p(4)  = S(1)*X(3) - S(4)*X(4)*X(5) - S(5)*X(5)^2;
    p(5)  = S(2);
    p(6)  = S(2)*X(1) + S(4);
    p(7)  = S(2)*X(2) + S(4)*X(4);
    p(8)  = S(2)*X(3) + S(4)*X(5);
    p(9)  = S(3);
    p(10) = S(3)*X(1) + S(5);
    p(11) = S(3)*X(2) + S(5)*X(4);
    p(12) = S(3)*X(3) + S(5)*X(5);
end