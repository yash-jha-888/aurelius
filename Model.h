#pragma once
#include <vector>
#include <Eigen/Dense>
#include "DenseLayer.h"

class Model
{
private:
    std::vector<DenseLayer> layers;

public:
    Model(const std::vector<int>& sizes);

    Eigen::MatrixXd forward(const Eigen::MatrixXd& X);
    
    void backward(const Eigen::MatrixXd& prediction, const Eigen::MatrixXd& target);
    
    void update(double learning_rate);
};