#pragma once

#include <Eigen/Dense>

double mse(const Eigen::MatrixXd& pred,
           const Eigen::MatrixXd& target);