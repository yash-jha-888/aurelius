#pragma once

#include <Eigen/Dense>
#include <memory>

class Initializer
{
public:
    virtual void initialize(Eigen::MatrixXd& W) const = 0;

    virtual std::unique_ptr<Initializer> clone() const = 0;

    virtual ~Initializer() = default;
};

class HeInitializer : public Initializer
{
public:
    void initialize(Eigen::MatrixXd& W) const override;

    std::unique_ptr<Initializer> clone() const override
    {
        return std::make_unique<HeInitializer>(*this);
    }
};

class XavierInitializer : public Initializer
{
public:
    void initialize(Eigen::MatrixXd& W) const override;

    std::unique_ptr<Initializer> clone() const override
    {
        return std::make_unique<XavierInitializer>(*this);
    }
};

class LeCunInitializer : public Initializer
{
public:
    void initialize(Eigen::MatrixXd& W) const override;

    std::unique_ptr<Initializer> clone() const override
    {
        return std::make_unique<LeCunInitializer>(*this);
    }
};