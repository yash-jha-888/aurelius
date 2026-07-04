#pragma once

#include <Eigen/Dense>

class DenseLayer
{
public:
    Eigen::MatrixXd W;
    Eigen::VectorXd b;

    Eigen::MatrixXd cached_input;
    Eigen::MatrixXd cached_Z;

    DenseLayer(int n_in, int n_out);

    Eigen::MatrixXd forward(const Eigen::MatrixXd& X);
};