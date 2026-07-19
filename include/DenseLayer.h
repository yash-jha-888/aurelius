#pragma once

#include <Eigen/Dense>
#include "Initializer.h"
class DenseLayer
{
public:
    Eigen::MatrixXd W;
    Eigen::VectorXd b;

    Eigen::MatrixXd cached_input;
    Eigen::MatrixXd cached_Z;

    Eigen::MatrixXd gradW;
    Eigen::VectorXd gradb;

    DenseLayer(
    int input_size,
    int output_size,
    const Initializer& init
    );

    Eigen::MatrixXd forward(const Eigen::MatrixXd& X);
};