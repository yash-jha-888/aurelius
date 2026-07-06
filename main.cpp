#include<iostream>
#include<Eigen/Dense>
#include<cmath>
#include"DenseLayer.h"
#include"Activation.h"
#include"Loss.h"
#include"Model.h"
#include"Dataset.h"
/*
Data layout - Row convention
One example - x is (1 x n_in) a row
A batch     - X is (batch x n_in) - each ROW is one example
Weights     - W is (n_in x n_out)
Forward     - Y = (X*W).rowwise() + b.transpose()   ->  (batch x n_out)
Each row of Y is one example's outputs.
*/    

 

int main(){
     
     // Load the MNIST dataset
     Eigen::MatrixXd X = load_images("data/train-images-idx3-ubyte");
     Eigen::MatrixXd Y = load_labels("data/train-labels-idx1-ubyte");
     std::cout << "loaded " << X.rows() << " images, " << Y.rows() << " labels\n";

     //Build the classifier model
     Model model({784, 128, 64, 10});
     double learning_rate = 0.1;
     int batch_size = 64;
     int num_samples = X.rows();

     // Train the model
     for (int epoch = 0; epoch < 30; epoch++) {
          double epoch_loss = 0.0;
          double epoch_accuracy = 0.0;
          int num_batches = 0;

          //mini=batch loop
          for (int start = 0; start + batch_size <= num_samples; start += batch_size) {
               //sliice the batch from the dataset
               Eigen::MatrixXd X_batch = X.middleRows(start, batch_size);
               Eigen::MatrixXd Y_batch = Y.middleRows(start, batch_size);

               // Forward pass -> softmax -> loss -> delta -> backward -> update
               Eigen::MatrixXd logits = model.forward(X_batch);
               Eigen::MatrixXd probs = softmax(logits);
               double acc = accuracy(probs, Y_batch);
               double loss = cross_entropy(probs, Y_batch);
               Eigen::MatrixXd delta = softmax_cross_entropy_delta(probs, Y_batch);
               model.backward(delta);
               model.update(learning_rate);

               epoch_loss += loss;
               epoch_accuracy += acc;
               num_batches++;
          }

          std::cout << "Epoch " << epoch
          << ", Avg. Loss: " << epoch_loss / num_batches
          << ", Accuracy: "
          << (epoch_accuracy / num_batches) * 100.0
          << "%"
          << std::endl;
     }

     
     return 0;
}
