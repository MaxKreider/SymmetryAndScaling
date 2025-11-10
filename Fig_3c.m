%% setup

%verified

%mr clean
clc
clf
clear
close all

%choose data type fixed equal spacing
data_type = 2;

%number of data points to consider ^2
NN = [56 80 113 160];

%number of trials over which to average
ntrials = 100;

%number of nearest neighbors to consider
K = 20;

%degree of local polynomial fit
ell = 3;

%number of GMLS iterations to perform
J = 2;

%domain
a = -1/2;
b = 1/2;

%true nullspace vector
truth = [0 1 -1 0 0 -1 1 0 0 0 0 0]';

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


        %% step 1 - generate data on the circle: u(t,x) = t + x

        %generate data
        if data_type == 1
            X= [];
            %fixed equally spaced values
            t = -1/2:2/N:1/2;
            [t,x] = meshgrid(t,t);
            X(:,1) = t(:);
            X(:,2) = x(:);

        else

            %uniformly distributed random samples in [a,b]^2
            X = (-1/2) + (1/2 - (-1/2)) * rand(N^2, 2);

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

        %P'*P
        A = zeros(12,12);
        for i = 1:size(X,1)

            % build small local rows for each S vector
            p1 = local_row(X(i,:), S1(:,i));
            p2 = local_row(X(i,:), S2(:,i));
            p3 = local_row(X(i,:), S3(:,i));

            % accumulate (P'*P)
            A = A + p1'*p1 + p2'*p2 + p3'*p3;
        end

        %take the svd to approximate nullspace
        [~,sv,V] = svd(A,'econ');

        %singular values
        sv = diag(sv);

        %approximation of the coefficients
        c= V(:,end-1);


        %% error analysis

        %compute the error
        local_err(qqq) = abs(sin(subspace(truth,c)));

    end

    %concatenate
    err_vec(aaa, :) = local_err;
end

%do the average
err_vec_avg = mean(err_vec);


%% plot error

%plot N
plot_N = NN.^2;

%theory
theory = (log(plot_N).^(ell/2).*plot_N.^(1-ell/2));

%visualize the error
figure(1)
loglog(plot_N,err_vec_avg,'k-','linewidth',3)
hold on
loglog(plot_N,err_vec_avg,'k.','markersize',40)
loglog(plot_N,theory*err_vec_avg(1)/theory(1),'r--','linewidth',3)
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
    p(2)  = S(1)*X(1) - S(4)*X(4);
    p(3)  = S(1)*X(2) - S(5)*X(4);
    p(4)  = S(1)*X(3) - S(4)*X(4)^2 - S(5)*X(4)*X(5);
    p(5)  = S(2);
    p(6)  = S(2)*X(1) - S(4)*X(5);
    p(7)  = S(2)*X(2) - S(5)*X(5);
    p(8)  = S(2)*X(3) - S(5)*X(5)^2 - S(4)*X(4)*X(5);
    p(9)  = S(3);
    p(10) = S(4) + S(3)*X(1);
    p(11) = S(5) + S(3)*X(2);
    p(12) = S(3)*X(3) + S(4)*X(4) + S(5)*X(5);
end


