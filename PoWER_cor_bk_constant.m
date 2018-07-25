% This file contains a sample implementation of PoWER
% using correlated constant exploration rate.
% Initialized by Jens Kober
% http://www-clmc.usc.edu/Resources/Software
% Modified by Li Shidi, Dec 2017
% https://github.com/MaruGreen/SAEPER

clear all;
close all;
clc;

% time step
dt = 0.005;
% length of the episode
T = 1;

% number of basis functions
n_rfs = 10;

ID = 1;
goal = 1;
% number of iterations
n_iter = 1200;
% beta_k
k = 1:n_iter+1;
beta_k = 3000 * ones(size(k)); % beta_k = constant
%beta_k = 300000 ./ (100+k);  % beta_k = f(k)

% initialize the motor primitive and storage variables
dcp('init',ID,n_rfs,'PoWER');

rt = 0:dt:T;

y = zeros(1,length(rt));
yd = zeros(1,length(rt));
ydd = zeros(1,length(rt));

Return = zeros(1,n_iter+1);
Q = zeros(length(rt),n_iter+1);
s_Return = zeros(n_iter+1,2);
param = zeros(n_rfs,n_iter+1);
variance = zeros(n_rfs,n_iter+1); % this one is to restore Lambda
basis = zeros(n_rfs,length(rt));
randn('state',20);

% set the initial variance
[Psi, Lambda, ~] = CreatePsi( n_rfs, beta_k(1) );
variance(:,1) = diag(Lambda);
variance(:,2) = diag(Lambda);

% start with all parameters set to zero
param(:,1) = zeros(n_rfs,1); 
current_param = param(:,1);

% apply the new parameters to the motor primitive
dcp('change',ID,'w',param(:,1));

% reset the motor primitive
dcp('reset_state',ID);
dcp('set_goal',ID,goal,1);

% run the motor primitive & precalculate the basis functions
for i=1:length(rt)
	% also store the values of the basis functions
    [y(i),yd(i),ydd(i),basis(:,i)] = dcp('run',ID,T,dt);
	% calculate the Q values
    Q(1:(end+1-i),1) = Q(1:(end+1-i),1) + exp(-(target(i)-y(i)).^2);
end
% normalize the Q values
Q(:,1) = Q(:,1)./length(rt);
% make Q become R
%Q(:,1) = Q(1,1)*ones(length(rt),1); 

% dcp('run') returns the unnormalized basis functions
basis = basis'./(sum(basis,1)'*ones(1,n_rfs));

% do the iterations
for iter=1:n_iter
    if (mod(iter,100)==0)
        disp(['Iter ', num2str(iter)]);
    end
    
    Return(iter) = Q(1,iter);

	% this lookup table will be used for the importance sampling
    s_Return(1,:) = [Return(iter) iter];
    s_Return = sortrows(s_Return);
    
    % update the policy parameters
    param_nom = zeros(n_rfs,1);
    param_dnom = zeros(n_rfs,n_rfs);

	% calculate the expectations (the normalization is taken care of by the division)
    % as importance sampling we take the 10 best rollouts
    for i=1:min(iter,10)
        % get the rollout number for the 10 best rollouts
        j = s_Return(end+1-i,2);
		% calulate W with the assumption that always only one basis functions is active
        temp_W = zeros(n_rfs,n_rfs,length(rt));
        for ii = 1 : length(rt)
            temp_W(:,:,ii) = basis(ii,:)'*basis(ii,:) / (basis(ii,:)*Psi*diag(variance(:,j))*Psi'*basis(ii,:)');
        end
		% calculate the exploration with respect to the current parameters
        % if you have time-varying exploration use 
        % temp_explore = (reshape(param(:,:,j),length(rt),n_rfs)-ones(length(rt),1)*current_param')';
        % instead
        temp_explore = param(:,j)-current_param;
		% repeat the Q values
        for ii = 1 : length(rt)
            param_nom = param_nom + temp_W(:,:,ii) * temp_explore * Q(ii,j);
            param_dnom = param_dnom + temp_W(:,:,ii) * Q(ii,j);
        end
    end
    
    % update the parameters
    param(:,iter+1) = current_param + (param_dnom) \ param_nom;
    
	% update the variances
    variance(:,iter+1) = variance(:,iter) * beta_k(iter+1) / beta_k(iter);
    
    % set the new mean of the parameters
    current_param = param(:,iter+1);

	% in the last rollout we want to get the return without exploration
    if iter~=n_iter
        temp = zeros(n_rfs,1);
        % create uncorrelated noise
        for j = 1:n_rfs
            temp(j) = normrnd(0,variance(j,iter+1).^0.5);
        end
        % transfer to correlated
        noisE = Psi * temp;
        %disp(noisE)
        param(:,iter+1) = param(:,iter+1) + noisE;
    end
    
    % apply the new parameters to the motor primitve
    dcp('change',ID,'w',param(:,iter+1));
    
    % reset the motor primitive
    dcp('reset_state',ID);
    dcp('set_goal',ID,goal,1);
    
    % run the motor primitive
    for i=1:length(rt)
        [y(i),yd(i),ydd(i)] = dcp('run',ID,T,dt);
		% calculate the Q values
        Q(1:(end+1-i),iter+1) = Q(1:(end+1-i),iter+1) + exp(-(target(i)-y(i)).^2);
    end
	% normalize the Q values
    Q(:,iter+1) = Q(:,iter+1)./length(rt);
    % make Q become R
    %Q(:,iter+1) = Q(1,iter+1)*ones(length(rt),1); 
end

% calculate the return of the final rollout
Return(iter+1) = Q(1,iter+1);

% show the performance of DMP
figure,
plot(y)
ylabel('position')
xlabel('time')
hold on,
plot(target(0:200))
hold off

% plot the return over the rollouts
figure,
plot(Return);
ylabel('return');
xlabel('rollouts');

disp(['Final Return ', num2str(Return(end))]);
save('cor_bk_constant.txt','-ascii','Return')



