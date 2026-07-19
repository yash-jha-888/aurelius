#include"Optimizer.h"

void SGD::update(
    Eigen::MatrixXd& W,
    Eigen::VectorXd& b,
    const Eigen::MatrixXd& dW,
    const Eigen::VectorXd& db,
    double learning_rate)
{
    W -= learning_rate * dW;
    b -= learning_rate * db;
}