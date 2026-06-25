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

int main(){
     
     DenseLayer layer1(3,4);
     DenseLayer layer2(4,2);

 
     MatrixXd X(2,3);

     X << -1,2,-7,
          4,-3,6;

     MatrixXd Z1 = layer1.forward(X);
     MatrixXd A1 = Z1.unaryExpr([](double v){return ReLU(v);});
     MatrixXd Z2 = layer2.forward(A1);
     MatrixXd output = Z2;

     
     cout << "Z1 outputs:\n" << Z1 << "\n\n";
     cout << "A1 outputs:\n" << A1 << "\n\n";
     cout << "Output:\n" << output << "\n";
     return 0;
}
