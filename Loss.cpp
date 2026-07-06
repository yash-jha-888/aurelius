#include "Loss.h"

double mse(const Eigen::MatrixXd& pred,
           const Eigen::MatrixXd& target)
{
    return (pred - target).array().square().mean();
}

Eigen::MatrixXd mse_delta(const Eigen::MatrixXd& pred,
                        const Eigen::MatrixXd& target)
{
    return (2.0 / pred.rows()) * (pred - target);
}

double cross_entropy(const Eigen::MatrixXd& pred,
                     const Eigen::MatrixXd& target)
{
    // Ensure numerical stability by adding a small epsilon
    const double epsilon = 1e-12;
    return -(target.array() * (pred.array() + epsilon).log()).sum() / pred.rows();
} 

Eigen::MatrixXd softmax_cross_entropy_delta(const Eigen::MatrixXd& probs,
                                          const Eigen::MatrixXd& target)
{
    return (probs - target) / probs.rows();
}