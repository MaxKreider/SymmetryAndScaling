%% setup

%verified

%mr clean
clc
clear

%choose data type (1 = fixed equal spacing, 2 = random)
data_type = 2;

%number of data points to consider
Num_count = 3;

%number of trials over which to average
ntrials = 20;

%number of nearest neighbors to consider
K = 40;

%degree of local polynomial fit
ell = 4;

%number of GMLS iterations to perform
J = 2;

%domain
a = 0*pi;
b = pi;

%initialize big data matrix
X_big = [];

%truth
truth = [1 0 0 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 -1 0 0 1 0]';

%error vector
err_vec = zeros(ntrials,Num_count);

%loop over multiple trials for each value of N
parfor aaa = 1:ntrials
    
    %do the parfor thing
    X_big = [];

    %initialize holder variable
    local_err = zeros(1, Num_count);

    %loop over the number of different samples
    for qqq = 1:Num_count

        %set number of samples
        if qqq == 1
            N = 225;
            nr = numel(1:1/3:1.3);
            ntheta = numel(0:1/15:2*pi); 
        elseif qqq == 2
            N = 300;
            nr = numel(1:0.25:1.3);
            ntheta = numel(0:0.05:2*pi);
        elseif qqq == 3
            N = 600;
            nr = numel(1:0.125:1.3);
            ntheta = numel(0:0.025:2*pi);
        end
        
        %create random variables
        r0 = 1 + (1.3 - 1) * rand(1, nr);
        theta0 = 0 + (2*pi - 0) * rand(1, ntheta);

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

        %P'*P
        A = zeros(12,12);
        for i = 1:N

            % build small local rows for each S vector
            p1 = local_row(X_big(i,:), S1(:,i));
            p2 = local_row(X_big(i,:), S2(:,i));

            % accumulate (P'*P)
            A = A + p1'*p1 + p2'*p2;
        end

        %take the svd to approximate nullspace
        [~,ss,V] = svd(A,'econ');

        sv = diag(ss)

        %approximation of the coefficients
        c= V(:,end);
        c1 = c
        c2 = V(:,end-1)

        %% error analysis

        %compute the error
        local_err(qqq) = abs(sin(subspace(truth,[c1,c2])));

    end

    %concatenate
    err_vec(aaa, :) = local_err;
    
end

%do the average
err_vec_avg = mean(err_vec);


%% plot error

%well
plot_N = [225*1*95,300*2*126,600*3*252];

%theory
theory = (log(plot_N).^(ell/3).*plot_N.^(1-ell/3));

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
    p(1) = S(1);
    p(2) = S(1)*X(1) - X(4)*S(4) - X(5)*S(5);
    p(3) = S(1)*X(2) - S(4)*X(4)^2 - S(5)*X(4)*X(5);
    p(4) = S(1)*X(3) - S(4)*X(4)*X(5) - S(5)*X(5)^2;
    p(5) = S(2);
    p(6) = S(2)*X(1) + S(4);
    p(7) = S(2)*X(2) + S(4)*X(4);
    p(8) = S(2)*X(3) + S(4)*X(5);
    p(9) = S(3);
    p(10) = S(3)*X(1) + S(5);
    p(11) = S(3)*X(2) + S(5)*X(4);
    p(12) = S(3)*X(3) + S(5)*X(5);
end
