#include "Loss.h"

double mse(const Eigen::MatrixXd& pred,
           const Eigen::MatrixXd& target)
{
    return (pred - target).array().square().mean();
}