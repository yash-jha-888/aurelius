#include<iostream>
#include<Eigen/Dense>
#include<cmath>
#include<limits>
#include <string>
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
     std::string data_dir = std::string(PROJECT_ROOT) + "/data/";
     Eigen::MatrixXd X = load_images(data_dir + "train-images-idx3-ubyte");
     Eigen::MatrixXd Y = load_labels(data_dir + "train-labels-idx1-ubyte");

     TrainValTestSplit data = train_val_test_split(X, Y, 0.1, 0.1);

     std::cout << "Train: " << data.X_train.rows() << '\n';

     std::cout << "Validation: " << data.X_val.rows() << '\n';

     std::cout << "Test: " << data.X_test.rows() << '\n';
     
     //Build the classifier model
     Model model({784, 128, 64, 10});

     Model best_model = model;

     double learning_rate = 0.1;
     int batch_size = 64;
     int num_samples = data.X_train.rows();

     double best_val_loss = std::numeric_limits<double>::infinity();
     int patience = 5;
     int patience_counter = 0;

     // Train the model
     for (int epoch = 0; epoch < 40; epoch++) {
          double epoch_loss = 0.0;
          double epoch_accuracy = 0.0;
          int num_batches = 0;

          //mini=batch loop
          for (int start = 0; start + batch_size <= num_samples; start += batch_size) {
               //sliice the batch from the dataset
               Eigen::MatrixXd X_batch = data.X_train.middleRows(start, batch_size);
               Eigen::MatrixXd Y_batch = data.Y_train.middleRows(start, batch_size);

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
          Eigen::MatrixXd val_logits = model.forward(data.X_val);
          Eigen::MatrixXd val_probs = softmax(val_logits);

          double val_loss = cross_entropy(val_probs, data.Y_val);
          const double min_delta = 1e-4;
          if (val_loss < best_val_loss - min_delta)
          {
              best_val_loss = val_loss;
              patience_counter = 0;

              best_model = model;
          }
          else
          {
              patience_counter++;
          }

          if (patience_counter >= patience)
          {
              std::cout << "\nEarly stopping at epoch " << epoch << "\n";
              std::cout << "Best validation loss = " << best_val_loss << "\n";
               break;
          }

          double val_acc = accuracy(val_probs, data.Y_val);

          std::cout << "Epoch " << epoch
          << ", Train Loss: " << epoch_loss / num_batches
          << ", Train Acc: " << (epoch_accuracy / num_batches) * 100 << "%"
          << ", Val Loss: " << val_loss
          << ", Val Acc: " << val_acc * 100 << "%"
          << std::endl;
     }

     model = best_model;
     std::cout << "Restored best checkpoint.\n";
     Eigen::MatrixXd test_logits = model.forward(data.X_test);
     Eigen::MatrixXd test_probs = softmax(test_logits);

     double test_loss = cross_entropy(test_probs, data.Y_test);
     double test_acc = accuracy(test_probs, data.Y_test);

     std::cout << "Test Loss: " << test_loss << ", Test Accuracy: " << test_acc * 100 << "%\n";

     return 0;
}
