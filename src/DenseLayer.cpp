#include "DenseLayer.h"
#include <random>
#include <cmath>

DenseLayer::DenseLayer(int n_in,
                       int n_out,
                       const Initializer& init)
{
    W.resize(n_in, n_out);
    init.initialize(W);

    b = Eigen::VectorXd::Zero(n_out);
    gradW = Eigen::MatrixXd::Zero(n_in,n_out);
    gradb = Eigen::VectorXd::Zero(n_out);
}

Eigen::MatrixXd DenseLayer::forward(const Eigen::MatrixXd& X)
{
    cached_input = X;
    cached_Z = (X * W).rowwise() + b.transpose();
    return cached_Z;
}