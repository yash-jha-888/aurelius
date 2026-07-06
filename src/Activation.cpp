#include "Activation.h"

double ReLU(double input_val){
     if(input_val<=0.0){
          return 0.0;
     }
     else{
          return input_val;          
     }
}

Eigen::MatrixXd softmax(const Eigen::MatrixXd& Z) {
    //max of each row
     Eigen::VectorXd rowMax = Z.rowwise().maxCoeff();
     //shift rows by subtracting rowMax
     Eigen::MatrixXd shiftedZ = Z.colwise() - rowMax;
     //exponentiate
     Eigen::MatrixXd expZ = shiftedZ.array().exp();
     //sum of exponentials for each row
     Eigen::VectorXd rowSum = expZ.rowwise().sum();
     //divide each row by the sum
     Eigen::MatrixXd probs = expZ.array().colwise() / rowSum.array();
     
     return probs;
}