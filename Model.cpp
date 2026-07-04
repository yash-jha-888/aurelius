#include "Model.h"
#include "Activation.h"
#include<Eigen/Dense>
Model::Model()
    : layer1(2,4),
      layer2(4,1)
{
}

Eigen::MatrixXd Model::forward(const Eigen::MatrixXd& X) {
     Eigen::MatrixXd Z1 = layer1.forward(X);
     Eigen::MatrixXd A1 = Z1.unaryExpr([](double v){return ReLU(v);});
     Eigen::MatrixXd Z2 = layer2.forward(A1);
     return Z2;
}

void Model::backward(const Eigen::MatrixXd& prediction, const Eigen::MatrixXd& target)
{
     const double n = prediction.rows(); 
     Eigen::MatrixXd delta2 = (2.0/n) * (prediction - target);  //n=1     
     layer2.gradW = layer2.cached_input.transpose() * delta2;
     layer2.gradb = delta2.colwise().sum();
     
     Eigen::MatrixXd delta_A1 = delta2 * layer2.W.transpose();   //blame on A1 (1x4)
     Eigen::MatrixXd relu_mask = (layer1.cached_Z.array() >0.0).cast<double>();  // 1 where Z1 > 0, else 0.
     Eigen::MatrixXd delta1 = delta_A1.array() * relu_mask.array(); //blame on Z1, hadamard product.
     layer1.gradW = layer1.cached_input.transpose() * delta1;
     layer1.gradb = delta1.colwise().sum();
}

void Model::update(double learning_rate)
{
     layer1.W -= learning_rate * layer1.gradW;
     layer2.W -= learning_rate * layer2.gradW;
     layer1.b -= learning_rate * layer1.gradb.transpose();
     layer2.b -= learning_rate * layer2.gradb.transpose();
}