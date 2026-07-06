#pragma once

#include <Eigen/Dense>

double mse(const Eigen::MatrixXd& pred,
           const Eigen::MatrixXd& target);

double cross_entropy(const Eigen::MatrixXd& pred,
                     const Eigen::MatrixXd& target);