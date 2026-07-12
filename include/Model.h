#pragma once
#include <vector>
#include <Eigen/Dense>
#include "DenseLayer.h"
#include <memory>
#include "Activation.h"

class Model
{
private:
    std::vector<DenseLayer> layers;
    std::unique_ptr<Activation> activation;

public:
    Model(const std::vector<int>& sizes,
          std::unique_ptr<Activation> act
        );

    Model(const Model& other);

    Model& operator=(const Model& other);

    Eigen::MatrixXd forward(const Eigen::MatrixXd& X);
    
    void backward(const Eigen::MatrixXd& output_delta);
    
    void update(double learning_rate);
};