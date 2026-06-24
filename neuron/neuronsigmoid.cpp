#include<iostream>
#include<Eigen/Dense>
#include<cmath>

using namespace std;
using namespace Eigen;

double sigmoid(double input_val){
     return 1.0 / (1.0 + exp(-input_val));
}

int main(){
 
    VectorXd weights(3);
    weights << 0.5,-0.5,1.0;

    VectorXd inputs(3);
    inputs << 1.0,-1.0,1.0;

    double bias = 0.1;
    
    double output = weights.dot(inputs) + bias;
    cout << sigmoid(output) << endl;

    return 0;
} 
