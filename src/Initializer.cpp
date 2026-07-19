#include "Initializer.h"

#include <random>
#include <cmath>

void HeInitializer::initialize(Eigen::MatrixXd& W) const
{
    static std::mt19937 generator(std::random_device{}());

    double stddev = std::sqrt(2.0 / W.rows());
    std::normal_distribution<double> dist(0.0, stddev);

    for (int i = 0; i < W.rows(); ++i) {
        for (int j = 0; j < W.cols(); ++j) {
            W(i, j) = dist(generator);
        }
    }
}

void XavierInitializer::initialize(Eigen::MatrixXd& W) const
{
    static std::mt19937 generator(std::random_device{}());

    double stddev = std::sqrt(2.0 / (W.rows() + W.cols()));

    std::normal_distribution<double> dist(0.0, stddev);

    for (int i = 0; i < W.rows(); ++i)
    {
        for (int j = 0; j < W.cols(); ++j)
        {
            W(i, j) = dist(generator);
        }
    }
}

void LeCunInitializer::initialize(Eigen::MatrixXd& W) const
{
    static std::mt19937 generator(std::random_device{}());

    double stddev = std::sqrt(1.0 / W.rows());

    std::normal_distribution<double> dist(0.0, stddev);

    for (int i = 0; i < W.rows(); ++i)
    {
        for (int j = 0; j < W.cols(); ++j)
        {
            W(i, j) = dist(generator);
        }
    }
}