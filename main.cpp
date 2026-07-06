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

     Eigen::MatrixXd pred(1, 3);
     pred << 0.7, 0.2, 0.1;
     Eigen::MatrixXd target(1, 3);
     target << 1.0, 0.0, 0.0;

     std::cout << cross_entropy(pred, target) << std::endl;     
     
     return 0;
}
