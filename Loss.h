#pragma once

#include <Eigen/Dense>

double mse(const Eigen::MatrixXd& pred,
           const Eigen::MatrixXd& target);

Eigen::MatrixXd mse_delta(const Eigen::MatrixXd& pred,
                        const Eigen::MatrixXd& target);

double cross_entropy(const Eigen::MatrixXd& pred,
                     const Eigen::MatrixXd& target);

Eigen::MatrixXd softmax_cross_entropy_delta(const Eigen::MatrixXd& probs,
                                          const Eigen::MatrixXd& target);