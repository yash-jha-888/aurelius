#include<iostream>
#include<Eigen/Dense>
#include<cmath>
#include"DenseLayer.h"
#include"Activation.h"
#include"Loss.h"
#include"Model.h"
/*
Data layout - Row convention
One example - x is (1 x n_in) a row
A batch     - X is (batch x n_in) - each ROW is one example
Weights     - W is (n_in x n_out)
Forward     - Y = (X*W).rowwise() + b.transpose()   ->  (batch x n_out)
Each row of Y is one example's outputs.
*/    



int main(){
     
     double learning_rate = 0.1;
     
     Model model;

 
     Eigen::MatrixXd X(4,2);
     X << 0, 0,
          0, 1,
          1, 0,
          1, 1;

     Eigen::MatrixXd target(4,1);
     target << 0,1,1,0;

     for (int iter = 0; iter < 10000; iter++) {

          Eigen::MatrixXd prediction = model.forward(X);
     
          double loss = mse(prediction, target);
     
          model.backward(prediction, target);

          model.update(learning_rate);

          if (iter % 500 == 0){
               std::cout << "iter " << iter << " loss: " << loss << "\n";
          }

     }

     std::cout << "Prediction :\n "<<model.forward(X) << "\n";

     return 0;
}
