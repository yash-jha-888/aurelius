#pragma once
#include <Eigen/Dense>
#include <string>

Eigen::MatrixXd load_images(const std::string& path);   // returns (num_images x 784), normalized [0,1]
Eigen::MatrixXd load_labels(const std::string& path);   // returns (num_labels x 10), one-hot