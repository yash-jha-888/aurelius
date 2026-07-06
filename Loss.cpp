#include "Loss.h"

double mse(const Eigen::MatrixXd& pred,
           const Eigen::MatrixXd& target)
{
    return (pred - target).array().square().mean();
}

double cross_entropy(const Eigen::MatrixXd& pred,
                     const Eigen::MatrixXd& target)
{
    // Ensure numerical stability by adding a small epsilon
    const double epsilon = 1e-12;
    return -(target.array() * (pred.array() + epsilon).log()).sum() / pred.rows();
} 