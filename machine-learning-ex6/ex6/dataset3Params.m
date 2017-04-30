function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

Citer = [0.01 0.03 0.1 0.3 1 3 10 30];
siter = [0.01 0.03 0.1 0.3 1 3 10 30];
minerror = 999999;

for Ci = 1:size(Citer,2),
	for sigmai = 1:size(siter,2),
		kerfun = @(x1, x2) (gaussianKernel(x1, x2, siter(sigmai)));
		model = svmTrain(X, y, Citer(Ci), kerfun, 1e-6, 20);
		predictions = svmPredict(model, Xval);
		errors = mean(double(predictions ~= yval));
		if errors<minerror
			minerror = errors;
			C = Citer(Ci);
			sigma = siter(sigmai);
		end;
	end;
end;







% =========================================================================

end
