#include "Model.h"
#include "Activation.h"
#include<Eigen/Dense>
Model::Model(const std::vector<int>& sizes)
{ 
     for (int i = 0; i < sizes.size() - 1; i++) {
        layers.push_back(DenseLayer(sizes[i], sizes[i+1]));
     }
}

Eigen::MatrixXd Model::forward(const Eigen::MatrixXd& X) {
    Eigen::MatrixXd current = X;
    for (int i = 0; i < layers.size(); i++) {
        current = layers[i].forward(current);         
        if (i < layers.size() - 1) {
            current = current.unaryExpr([](double v){ return ReLU(v); });
        }
    }
    return current;
}

void Model::backward(const Eigen::MatrixXd& output_delta)
{
     const double n = output_delta.rows(); 
     Eigen::MatrixXd delta = output_delta;
     for (int i = layers.size() - 1; i >= 0; i--) {
         if (i < layers.size() - 1) {
             Eigen::MatrixXd relu_mask = (layers[i].cached_Z.array() > 0.0).cast<double>();
             delta = (delta * layers[i + 1].W.transpose()).array() * relu_mask.array();
         }
         layers[i].gradW = layers[i].cached_input.transpose() * delta;
         layers[i].gradb = delta.colwise().sum();
     }
}

void Model::update(double learning_rate)
{
     for (auto& layer : layers) {
         layer.W -= learning_rate * layer.gradW;
         layer.b -= learning_rate * layer.gradb.transpose();
     }
}