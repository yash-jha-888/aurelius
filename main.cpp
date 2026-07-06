#include<iostream>
#include<Eigen/Dense>
#include<cmath>
#include"DenseLayer.h"
#include"Activation.h"
#include"Loss.h"
#include"Model.h"
#include"Dataset.h"
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
     /*   
     Model model({784, 128, 64, 10});     // <-- sizes list, matches new constructor
     
     Eigen::MatrixXd X = load_images("data/train-images-idx3-ubyte");
     Eigen::MatrixXd Y = load_labels("data/train-labels-idx1-ubyte");

     std::cout << "loaded " << X.rows() << " images, " << Y.rows() << " labels\n";
     */

     Eigen::MatrixXd X(4, 2);
     X <<  0, 0,
           0, 1,
           1, 0,
           1, 1;
     Eigen::MatrixXd Y(4, 1);
     Y << 0,
          1,
          1,
          0;
     
     Model model({2, 4, 1}); // 2 input features, 4 hidden neurons, 1 output neuron
     for (int epoch = 0; epoch < 10000; ++epoch) {
         Eigen::MatrixXd predictions = model.forward(X);
         double loss = mse(predictions, Y);
         Eigen::MatrixXd delta = mse_delta(predictions, Y);
         model.backward(delta);
         model.update(learning_rate);

         if (epoch % 1000 == 0) {
             std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
         }
     }
     std::cout << "Final predictions:\n" << model.forward(X) << std::endl;
     
     return 0;
}
