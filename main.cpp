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
         MatrixXd cached_Z;

         DenseLayer( int n_in, int n_out){
          W = MatrixXd::Random(n_in, n_out);
          b = VectorXd::Zero(n_out);
         }
         MatrixXd forward(const MatrixXd& X) { 
          cached_input=X;
          cached_Z = (X*W).rowwise() + b.transpose();
          return cached_Z;
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

 
     MatrixXd X(4,2);
     X << 0, 0,
          0, 1,
          1, 0,
          1, 1;

     MatrixXd target(4,1);
     target << 0,1,1,0;
     float n=X.rows();

     MatrixXd gradW1 = MatrixXd::Zero(layer1.W.rows(),layer1.W.cols());
     MatrixXd gradW2 = MatrixXd::Zero(layer2.W.rows(),layer2.W.cols());
     MatrixXd gradb1 = VectorXd::Zero(layer1.b.size());
     MatrixXd gradb2 = VectorXd::Zero(layer2.b.size());



     for (int iter = 0; iter < 10000; iter++) {

     MatrixXd prediction = forward_pass(X, layer1, layer2);
     double loss = mse(prediction, target);

     //BACKWARD

     MatrixXd delta2 = (2.0/n) * (prediction - target);  //n=1     
     MatrixXd gradW2 = layer2.cached_input.transpose() * delta2;
     MatrixXd gradb2 = delta2.colwise().sum();
     
     MatrixXd delta_A1 = delta2 * layer2.W.transpose();   //blame on A1 (1x4)
     MatrixXd relu_mask = (layer1.cached_Z.array() >0.0).cast<double>();  // 1 where Z1 > 0, else 0.
     MatrixXd delta1 = delta_A1.array() * relu_mask.array(); //blame on Z1, hadamard product.
     MatrixXd gradW1 = layer1.cached_input.transpose() * delta1;
     MatrixXd gradb1 = delta1.colwise().sum();
     

     //UPDATE
     layer1.W -= learning_rate * gradW1;
     layer2.W -= learning_rate * gradW2;
     layer1.b -= learning_rate * gradb1.transpose();
     layer2.b -= learning_rate * gradb2.transpose();

     if (iter % 500 == 0){
          cout << "iter " << iter << " loss: " << loss << "\n";
     }

     }

     cout << "Prediction :\n "<<forward_pass(X, layer1, layer2) << "\n";

     return 0;
}
