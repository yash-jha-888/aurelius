#include "DenseLayer.h"
#include <random>
#include <cmath>

DenseLayer::DenseLayer(int n_in, int n_out)
{
    static std::mt19937 generator(std::random_device{}());

    double stddev = std::sqrt(2.0 / n_in);
    std::normal_distribution<double> dist(0.0, stddev);

    W.resize(n_in, n_out);
    for (int i = 0; i < W.rows(); ++i) {
        for (int j = 0; j < W.cols(); ++j) {
            W(i, j) = dist(generator);
        }
    }
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