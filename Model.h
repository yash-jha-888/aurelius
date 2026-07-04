#pragma once

#include <Eigen/Dense>
#include "DenseLayer.h"

class Model
{
private:
    DenseLayer layer1;
    DenseLayer layer2;

public:
    Model();

    Eigen::MatrixXd forward(const Eigen::MatrixXd& X);
    
    void backward(const Eigen::MatrixXd& prediction, const Eigen::MatrixXd& target);
    
    void update(double learning_rate);
};