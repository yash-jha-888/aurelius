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

     double loss_before = mse(forward_pass(X, layer1, layer2), target);

     cout << "Loss before training: " << loss_before << endl;



     return 0;
}
