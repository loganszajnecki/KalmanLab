#ifndef KALMANFILTER_H
#define KALMANFILTER_H

#include <string>
#include <Eigen/Dense>
using namespace Eigen;

class KalmanFilter {
    public:
        // Default constructor
        KalmanFilter();

        // Overload constructor
        KalmanFilter(MatrixXd A, MatrixXd B, MatrixXd C,
                     MatrixXd Q, MatrixXd R, MatrixXd P0,
                     MatrixXd x0, unsigned int maxSimulationSamples);

        void updateEstimate(MatrixXd measurement);

        void predictEstimate(MatrixXd externalInput); // Step number 1

        static MatrixXd openData(std::string fileToOpen);

        void saveData(std::string estimatesAposterioriFile, std::string estimatesAprioriFile, 
                      std::string covarianceAposterioriFile, std::string covarianceAprioriFile, 
                      std::string gainMatricesFile, std::string errorsFile) const;

    private:
        // Track current time step of the estimator
        unsigned int k;

        // m - input dimension, n - state dimension, r - output dimension
        unsigned int m, n, r;

        // Matrices of unspecified dimension
        MatrixXd A, B, C, Q, R, P0;
        MatrixXd x0; // Initial state

        // A posteriori state estimate (columnwise) 
        MatrixXd estimatesAposteriori; // [xhat_0^+, xhat_1^+, xhat_2^+, ..., xhat_N^+]

        // A priori state estimate
        MatrixXd estimatesApriori; // [xhat_1^-, ... , xhat_N^-]

        // A posteriori covariance
        MatrixXd covarianceAposteriori; // [P_0^+, P_1^+, ... , P_N^+]

        //  A priori covariance
        MatrixXd covarianceApriori; // [P_1^-, P_2^-, ... , P_N^-]

        // Kalman Gains
        MatrixXd gainMatrices; // [K_1, K_2, ... , K_N]

        // Prediction errors, error_k = y_k - C * x_k^{-}
        MatrixXd errors; // [e_1, e_2, ... , e_N]
};


#endif // KALMANFILTER_H
