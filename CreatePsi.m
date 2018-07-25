function [Psi, Lambda, Sigma] = CreatePsi( order, beta )
% This is the function to create the matrix R to generate correlated noise
% Sigma = beta * R^{-1}
% Lambda = Psi * Sigma * Psi^{-1}

% create A
A = -2 * eye(order);
for i = 1 : order-1
    A(i,i+1) = 1;
    A(i+1,i) = 1;
end
A = [1 zeros(1,order-1); A; zeros(1,order-1) 1];
% create R
R = A' * A;
% create Sigma
Sigma = inv(R / beta);
% eigen-decomposition for Sigma
[Psi, Lambda] = eig(Sigma);

end

