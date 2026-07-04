#include<iostream>
#include<Eigen/Dense>
#include<cmath>
#include"DenseLayer.h"
#include"Activation.h"
#include"Loss.h"

/*
Data layout - Row convention
One example - x is (1 x n_in) a row
A batch     - X is (batch x n_in) - each ROW is one example
Weights     - W is (n_in x n_out)
Forward     - Y = (X*W).rowwise() + b.transpose()   ->  (batch x n_out)
Each row of Y is one example's outputs.
*/    

Eigen::MatrixXd forward_pass(const Eigen::MatrixXd& X, DenseLayer& dense1, DenseLayer& dense2) {
     Eigen::MatrixXd Z1 = dense1.forward(X);
     Eigen::MatrixXd A1 = Z1.unaryExpr([](double v){return ReLU(v);});
     Eigen::MatrixXd Z2 = dense2.forward(A1);
     return Z2;
} 

int main(){
     
     double epsilon = 1e-5;
     double learning_rate = 0.1;
     
     DenseLayer layer1(2,4);
     DenseLayer layer2(4,1);

 
     Eigen::MatrixXd X(4,2);
     X << 0, 0,
          0, 1,
          1, 0,
          1, 1;

     Eigen::MatrixXd target(4,1);
     target << 0,1,1,0;
     const double n=X.rows();

     Eigen::MatrixXd gradW1 = Eigen::MatrixXd::Zero(layer1.W.rows(),layer1.W.cols());
     Eigen::MatrixXd gradW2 = Eigen::MatrixXd::Zero(layer2.W.rows(),layer2.W.cols());
     Eigen::MatrixXd gradb1 = Eigen::VectorXd::Zero(layer1.b.size());
     Eigen::MatrixXd gradb2 = Eigen::VectorXd::Zero(layer2.b.size());



     for (int iter = 0; iter < 10000; iter++) {

     Eigen::MatrixXd prediction = forward_pass(X, layer1, layer2);
     double loss = mse(prediction, target);

     //BACKWARD

     Eigen::MatrixXd delta2 = (2.0/n) * (prediction - target);  //n=1     
     Eigen::MatrixXd gradW2 = layer2.cached_input.transpose() * delta2;
     Eigen::MatrixXd gradb2 = delta2.colwise().sum();
     
     Eigen::MatrixXd delta_A1 = delta2 * layer2.W.transpose();   //blame on A1 (1x4)
     Eigen::MatrixXd relu_mask = (layer1.cached_Z.array() >0.0).cast<double>();  // 1 where Z1 > 0, else 0.
     Eigen::MatrixXd delta1 = delta_A1.array() * relu_mask.array(); //blame on Z1, hadamard product.
     Eigen::MatrixXd gradW1 = layer1.cached_input.transpose() * delta1;
     Eigen::MatrixXd gradb1 = delta1.colwise().sum();
     

     //UPDATE
     layer1.W -= learning_rate * gradW1;
     layer2.W -= learning_rate * gradW2;
     layer1.b -= learning_rate * gradb1.transpose();
     layer2.b -= learning_rate * gradb2.transpose();

     if (iter % 500 == 0){
          std::cout << "iter " << iter << " loss: " << loss << "\n";
     }

     }

     std::cout << "Prediction :\n "<<forward_pass(X, layer1, layer2) << "\n";

     return 0;
}
