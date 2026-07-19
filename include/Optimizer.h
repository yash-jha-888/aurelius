#pragma once

#include <Eigen/Dense>
#include <memory>


class Optimizer
{
public:
    virtual void update(
        Eigen::MatrixXd& W,
        Eigen::VectorXd& b,
        const Eigen::MatrixXd& dW,
        const Eigen::VectorXd& db,
        double learning_rate) = 0;

    virtual std::unique_ptr<Optimizer> clone() const = 0;

    virtual ~Optimizer() = default;
};

class SGD : public Optimizer
{
    public:
        void update(
            Eigen::MatrixXd& W,
            Eigen::VectorXd& b,
            const Eigen::MatrixXd& dW,
            const Eigen::VectorXd& db,
            double learning_rate) override;

        std::unique_ptr<Optimizer> clone() const override
        {
            return std::make_unique<SGD>(*this);
        }
};