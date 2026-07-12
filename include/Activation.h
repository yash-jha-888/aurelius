#pragma once

#include <Eigen/Dense>
#include <memory>

class Activation
{
public:

    virtual Eigen::MatrixXd forward(
        const Eigen::MatrixXd& Z) const = 0;

    virtual Eigen::MatrixXd derivative(
        const Eigen::MatrixXd& Z) const = 0;
    
    virtual std::unique_ptr<Activation> clone() const = 0;

    virtual ~Activation() = default;
};

class ReLU : public Activation
{
public:

    Eigen::MatrixXd forward(
        const Eigen::MatrixXd& Z) const override;

    Eigen::MatrixXd derivative(
        const Eigen::MatrixXd& Z) const override;

    std::unique_ptr<Activation> clone() const override
    {
        return std::make_unique<ReLU>(*this);
    }
};

class LeakyReLU : public Activation
{
public:

    Eigen::MatrixXd forward(
        const Eigen::MatrixXd& Z) const override;

    Eigen::MatrixXd derivative(
        const Eigen::MatrixXd& Z) const override;
    
    std::unique_ptr<Activation> clone() const override
    {
        return std::make_unique<LeakyReLU>(*this);
    }
};

Eigen::MatrixXd softmax(const Eigen::MatrixXd& Z);