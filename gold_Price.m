clear;
clc;

%test data load
data = xlsread('data.xlsx');
dollerToEuroData = data(:,1)/max(data(:,1));
oilPricesData = data(:,2)/max(data(:,2));
goldPricesData = data(:,3)/max(data(:,3));

dollerToEuroTrain = dollerToEuroData(1:809,1);
oilPricesTrain = oilPricesData(1:809,1);
goldPricesTrain = goldPricesData(1:809,1);

dollerToEuroTest = dollerToEuroData(810:951,1);
oilPricesTest = oilPricesData(810:951,1);
goldPricesTest = goldPricesData(810:951,1);

normalizeGraph = max(data(:,3));
N = length(dollerToEuroTrain);
% Initializing the weights matrices W1 & W2

    for b = 1:2
        W2(1, b) = rand();
    end
    
    for v = 1:2
    for s = 1:2
        W1(v, s) = rand();
    end
    end

% Training patterns
training_sets = [dollerToEuroTrain oilPricesTrain]';

% Initializing bias values
b1 = [rand(), rand()];
b2 = [rand()];

%-----------------------------------------------Training----------------------------------------------------------------------
for q = 1:700;
    
    % Total error
    Err = 0.0;
    
    Eot  = 0;
    %For each training pattern
    for z = 1:N
        
        % Output values of input layer
        for k = 1:2
            oi(k,1) = -training_sets(k,z);
        end
        
        % input values of hidden layer
        ih = W1 * oi + b1';
        
        % Output values of hidden layer
        for a = 1:2
            oh(a, 1) = 1 / (1 + exp(-ih(a)));
        end
        
        % input values of output layer
        io = W2 * oh + b2';
        
        % output values of output layer
        oo = 1 / (1 + exp(-io));
       
         %Total error of each pattern
        Eo = (1/2)*(goldPricesTrain(z)-oo)^2;
        Eot = Eot + Eo;
       
        
         % ---- Back-Propagation ----
         delta = (oo - goldPricesTrain(z)) * oo * (1 - oo);
         
         % Calculating the delta weights of W2
          for c = 1:2
            delta_w2(:, c) = delta * oh(c);
          end
     
          % Calculating the delta weights of W1
          for f = 1:2

            for g = 1:2
                delta_w1(f, g) = delta * W2(:, f) * oh(f) * (1 - oh(f)) * oi(g);
            end

          end
         
          % Calculating the delta weights of bias 1
          for bias_1_w_num = 1:2
            delta_b1(bias_1_w_num) = delta * W2(:, bias_1_w_num) * oh(bias_1_w_num) * (1 - oh(bias_1_w_num));
          end
          
          % Learning rate
          learning_rate = 3;

          % Updating W2, W1, bias 1, and bias 2
          W2 = W2 - learning_rate * delta_w2;
          W1 = W1 - learning_rate * delta_w1;
          b2 = b2 - learning_rate * delta;
          b1 = b1 - learning_rate * delta_b1;
    end
  
    % Collecting each total error of each time of training
    Err = Err + Eot;
    Err_list(q) = Err;
end

%changes of error
figure(1)
plot(Err_list);
title('changes on error after each iteration');
xlabel('Times of Iterations');
ylabel('Total Error');


%---------------------------------------------------Testing--------------------------------------------------------
%Testing pattern
testing_sets = [dollerToEuroTest oilPricesTest]';
M = length(dollerToEuroTest);

 for z = 1:M
        
        % Output values of input layer
        for k = 1:2
            oi(k,1) = -testing_sets(k,z);
        end
        
        % input values of hidden layer
        ih = W1 * oi + b1';
        
        % Output values of hidden layer
        for a = 1:2
            oh(a, 1) = 1 / (1 + exp(-ih(a)));
        end
        
        % input values of output layer
        io = W2 * oh + b2';
        
        % output values of output layer
        oo = 1 / (1 + exp(-io));
        
        biased_result(:, z) = oo;
end
biased_result = biased_result'*normalizeGraph;
goldPricesTest = goldPricesTest*normalizeGraph;
% Output the test result
figure(2)
hold on

plot(1:142,biased_result);
plot( 1:142,goldPricesTest);
legend('Predicted Price','Actual Price');
title('Test Output');
xlabel('Date'); 
ylabel('price');
set(gca,'XTick',[0 36 72 107 142]);   
set(gca,'XTickLabel',{'5/01/2020','6/6/2020','7/13/2020','8/16/2020','9/20/2020'});
hold off





















