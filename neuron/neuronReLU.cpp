#include<iostream>
#include<Eigen/Dense>

using namespace std;
using namespace Eigen;

double ReLu(double input_val){
     if(input_val<=0.0){
          return 0.0;
     }
     else{
          return input_val;          
     }
}

int main(){
 
    VectorXd weights(3);

    weights << 0.5,-0.5,1.0;

    VectorXd inputs(3);

    inputs << 1.0,-1.0,1.0;

    double bias = 0.1;
    
    double output = weights.dot(inputs) + bias;
    cout << ReLu(output) << endl;

    return 0;
}
