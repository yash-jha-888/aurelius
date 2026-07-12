#include "Activation.h"


// ReLU


Eigen::MatrixXd ReLU::forward(const Eigen::MatrixXd& Z) const
{
    return Z.unaryExpr([](double z)
    {
        return z > 0.0 ? z : 0.0;
    });
}

Eigen::MatrixXd ReLU::derivative(const Eigen::MatrixXd& Z) const
{
    return (Z.array() > 0.0).cast<double>();
}


// LeakyReLU


Eigen::MatrixXd LeakyReLU::forward(const Eigen::MatrixXd& Z) const
{
    return Z.unaryExpr([](double z)
    {
        return z > 0.0 ? z : 0.01 * z;
    });
}

Eigen::MatrixXd LeakyReLU::derivative(const Eigen::MatrixXd& Z) const
{
    return Z.unaryExpr([](double z)
    {
        return z > 0.0 ? 1.0 : 0.01;
    });
}


// Softmax


Eigen::MatrixXd softmax(const Eigen::MatrixXd& Z)
{
    Eigen::VectorXd rowMax = Z.rowwise().maxCoeff();

    Eigen::MatrixXd shiftedZ = Z.colwise() - rowMax;

    Eigen::MatrixXd expZ = shiftedZ.array().exp();

    Eigen::VectorXd rowSum = expZ.rowwise().sum();

    Eigen::MatrixXd probs =
        expZ.array().colwise() / rowSum.array();

    return probs;
}