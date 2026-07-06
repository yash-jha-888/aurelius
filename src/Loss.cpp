#include "Loss.h"

double mse(const Eigen::MatrixXd& pred,
           const Eigen::MatrixXd& target)
{
    return (pred - target).array().square().mean();
}

Eigen::MatrixXd mse_delta(const Eigen::MatrixXd& pred,
                        const Eigen::MatrixXd& target)
{
    return (2.0 / pred.rows()) * (pred - target);
}

double cross_entropy(const Eigen::MatrixXd& pred,
                     const Eigen::MatrixXd& target)
{
    // Ensure numerical stability by adding a small epsilon
    const double epsilon = 1e-12;
    return -(target.array() * (pred.array() + epsilon).log()).sum() / pred.rows();
} 

Eigen::MatrixXd softmax_cross_entropy_delta(const Eigen::MatrixXd& probs,
                                          const Eigen::MatrixXd& target)
{
    return (probs - target) / probs.rows();
}

double accuracy(const Eigen::MatrixXd& probs,
                const Eigen::MatrixXd& target)
{
    int correct = 0;

    for(int i = 0; i < probs.rows(); i++)
    {
        Eigen::Index pred_idx;
        Eigen::Index target_idx;

        probs.row(i).maxCoeff(&pred_idx);
        target.row(i).maxCoeff(&target_idx);

        if(pred_idx == target_idx)
            correct++;
    }

    return static_cast<double>(correct) / probs.rows();
}