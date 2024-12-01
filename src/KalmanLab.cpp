#include<iostream>
// these two files define the Kalman filter class
#include "KalmanFilter.h"

using namespace Eigen;

int main() {
    // time step
    double h = 0.1;
    double noiseStd = 0.5;

    Matrix <double,3,3> A {{1,h,0.5*(h*h)} , {0,1,h} , {0,0,1}};
    Matrix <double,3,1> B {{0},{0},{0}}; 
    Matrix <double,1,3> C {{1,0,0}};
    // covariance matrix of the state estimation error 
    MatrixXd P0; 
    P0.resize(3,3); 
    P0= MatrixXd::Identity(3,3);

    // covariance matrix of the measurement noise
    Matrix <double,1,1> R {{noiseStd*noiseStd}};

    // covariance matrix of the state disturbance
    MatrixXd Q; Q.resize(3,3); Q.setZero(3,3);

    // guess of the initial state estimate
    Matrix <double,3,1> x0 {{0},{0},{0}};

    MatrixXd outputNoisy;
    // you can call the function openData() without creating the object of the class Kalman filter
    // since openData is a static function!
    outputNoisy=KalmanFilter::openData("/home/szajnecki/Projects/KalmanLab/noisySignal.csv");

    int sampleNumber=outputNoisy.rows();

    unsigned int maxDataSamples=sampleNumber+10;

    KalmanFilter KalmanFilterObject(A, B, C, Q, R, P0, x0, maxDataSamples);
    MatrixXd inputU, outputY;
    inputU.resize(1,1);
    outputY.resize(1,1);
 
    // this is the main Kalman filter loop - predict and update
    for (int index1=0; index1<sampleNumber; index1++)
    {
        inputU(0,0)=0;
        outputY(0,0)=outputNoisy(index1,0);
        // predict the estimate
        KalmanFilterObject.predictEstimate(inputU);
        // update the estimate
        KalmanFilterObject.updateEstimate(outputY);
    }

    // save the data
    KalmanFilterObject.saveData("estimatesAposteriori.csv", "estimatesAprioriFile.csv", 
                                "covarianceAposterioriFile.csv", "covarianceAprioriFile.csv", 
                                "gainMatricesFile.csv", "errorsFile.csv");

    return 0;
}