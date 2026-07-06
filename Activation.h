#pragma once
#include <Eigen/Dense>
double ReLU(double input_val);

Eigen::MatrixXd softmax(const Eigen::MatrixXd& Z);