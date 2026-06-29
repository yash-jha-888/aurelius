#include<iostream>
#include<Eigen/Dense>
#include<cmath>

/*
Data layout - Row convention
One example - x is (1 x n_in) a row
A batch     - X is (batch x n_in) - each ROW is one example
Weights     - W is (n_in x n_out)
Forward     - Y = (X*W).rowwise() + b.transpose()   ->  (batch x n_out)
Each row of Y is one example's outputs.
*/    

using namespace std;
using namespace Eigen;

double ReLU(double input_val){
     if(input_val<=0.0){
          return 0.0;
     }
     else{
          return input_val;          
     }
}

double mse(const MatrixXd& pred, const MatrixXd& target){
    return (pred - target).array().square().mean();

}


class DenseLayer {
     public:
         MatrixXd W;
         VectorXd b;
         MatrixXd cached_input;

         DenseLayer( int n_in, int n_out){
          W = MatrixXd::Random(n_in, n_out);
          b = VectorXd::Zero(n_out);
         }
         MatrixXd forward(const MatrixXd& X) { 
          cached_input=X;
          return (X*W).rowwise() + b.transpose();

         }
};



MatrixXd forward_pass(const MatrixXd& X, DenseLayer& dense1, DenseLayer& dense2) {
     MatrixXd Z1 = dense1.forward(X);
     MatrixXd A1 = Z1.unaryExpr([](double v){return ReLU(v);});
     MatrixXd Z2 = dense2.forward(A1);
     return Z2;
} 



int main(){
     
     double epsilon = 1e-5;
     double learning_rate = 0.1;
     
     DenseLayer layer1(2,4);
     DenseLayer layer2(4,1);

 
     MatrixXd X(1,2);
     X << 0, 1;

     MatrixXd target(1,1);
     target << 1;


     MatrixXd gradW1 = MatrixXd::Zero(layer1.W.rows(),layer1.W.cols());
     MatrixXd gradW2 = MatrixXd::Zero(layer2.W.rows(),layer2.W.cols());
     MatrixXd gradb1 = VectorXd::Zero(layer1.b.size());
     MatrixXd gradb2 = VectorXd::Zero(layer2.b.size());



     for (int iter = 0; iter < 1000; iter++) {
     double loss_before = mse(forward_pass(X, layer1, layer2), target);
     for(int i = 0; i < layer1.W.rows(); i++){
          for ( int j = 0; j < layer1.W.cols(); j++){
               double original = layer1.W(i, j);
               layer1.W(i, j)+=epsilon;
               double loss_after = mse(forward_pass(X, layer1, layer2), target);
               gradW1(i, j) = (loss_after - loss_before)/epsilon;
               layer1.W(i, j)=original;
          }
     }
     for(int i = 0; i < layer2.W.rows(); i++){
          for ( int j = 0; j < layer2.W.cols(); j++){
               double original = layer2.W(i, j);
               layer2.W(i, j)+=epsilon;
               double loss_after = mse(forward_pass(X, layer1, layer2), target);
               gradW2(i, j) = (loss_after - loss_before)/epsilon;
               layer2.W(i, j)=original;
          }
     }
     for(int i = 0; i < layer1.b.size(); i++){
               double original = layer1.b(i);
               layer1.b(i)+=epsilon;
               double loss_after = mse(forward_pass(X, layer1, layer2), target);
               gradb1(i) = (loss_after - loss_before)/epsilon;
               layer1.b(i)=original;
          }
     for(int i = 0; i < layer2.b.size(); i++){
               double original = layer2.b(i);
               layer2.b(i)+=epsilon;
               double loss_after = mse(forward_pass(X, layer1, layer2), target);
               gradb2(i) = (loss_after - loss_before)/epsilon;
               layer2.b(i)=original;
          }
     layer1.W -= learning_rate * gradW1;
     layer2.W -= learning_rate * gradW2;
     layer1.b -= learning_rate * gradb1;
     layer2.b -= learning_rate * gradb2;

     if (iter % 100 == 0){
          cout << "iter " << iter << " loss: " << loss_before << "\n";
     }

     }

     cout << "Prediction : "<<forward_pass(X, layer1, layer2) << "\n";
     

     return 0;
}
