#include "DenseLayer.h"

DenseLayer::DenseLayer(int n_in, int n_out)
{
    W = Eigen::MatrixXd::Random(n_in, n_out);
    b = Eigen::VectorXd::Zero(n_out);
}

Eigen::MatrixXd DenseLayer::forward(const Eigen::MatrixXd& X)
{
    cached_input = X;
    cached_Z = (X * W).rowwise() + b.transpose();
    return cached_Z;
}