#include "Activation.h"

double ReLU(double input_val){
     if(input_val<=0.0){
          return 0.0;
     }
     else{
          return input_val;          
     }
}