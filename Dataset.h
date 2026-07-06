#pragma once
#include <Eigen/Dense>
#include <string>

Eigen::MatrixXd load_images(const std::string& path);   // returns (num_images x 784), normalized [0,1]
Eigen::MatrixXd load_labels(const std::string& path);   // returns (num_labels x 10), one-hot

struct ShuffledData
{
    Eigen::MatrixXd X;
    Eigen::MatrixXd Y;
};

ShuffledData shuffle_data(const Eigen::MatrixXd& X,
                          const Eigen::MatrixXd& Y,
                          unsigned int seed = 42);

struct DataSplit
{
    Eigen::MatrixXd X_a, Y_a;
    Eigen::MatrixXd X_b, Y_b;
};

DataSplit split_data(const Eigen::MatrixXd& X,
                     const Eigen::MatrixXd& Y,
                     double fraction_b = 0.1);