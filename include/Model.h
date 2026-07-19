#pragma once
#include <vector>
#include <Eigen/Dense>
#include "DenseLayer.h"
#include <memory>
#include "Activation.h"
#include "Initializer.h"
#include "Optimizer.h"

class Model
{
private:
    std::vector<DenseLayer> layers;
    std::unique_ptr<Activation> activation;
    std::unique_ptr<Initializer> initializer;
    std::unique_ptr<Optimizer> optimizer;

public:
    Model(const std::vector<int>& sizes,
          std::unique_ptr<Activation> act,
          std::unique_ptr<Initializer> init,
          std::unique_ptr<Optimizer> opt
        );

    Model(const Model& other);

    Model& operator=(const Model& other);

    Eigen::MatrixXd forward(const Eigen::MatrixXd& X);
    
    void backward(const Eigen::MatrixXd& output_delta);
    
    void update(double learning_rate);
};