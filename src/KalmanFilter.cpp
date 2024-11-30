#include <iostream>
#include <tuple>
#include <string>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include "KalmanFilter.h"

using namespace Eigen;

KalmanFilter::KalmanFilter() {}

KalmanFilter::KalmanFilter(MatrixXd Ainput, MatrixXd Binput, MatrixXd Cinput,
                           MatrixXd Qinput, MatrixXd Rinput, MatrixXd P0input,
                           MatrixXd x0input, unsigned int maxSimulationSamples) {
    k = 0;
    A = Ainput; B = Binput; C = Cinput; Q = Qinput;
    R = Rinput; P0 = P0input; x0 = x0input;

    // Extract dimensions
    n = A.rows(); m = B.cols(); r = C.rows();

    // Assign matrix sizes
    estimatesAposteriori.resize(n, maxSimulationSamples);
    estimatesAposteriori.setZero();
    estimatesAposteriori.col(0) = x0;

    estimatesApriori.resize(n, maxSimulationSamples);
    estimatesApriori.setZero();

    covarianceAposteriori.resize(n, n*maxSimulationSamples);
    covarianceAposteriori.setZero();
    covarianceAposteriori(all,seq(0,n-1))=P0;
 
    covarianceApriori.resize(n,n*maxSimulationSamples);
    covarianceApriori.setZero();

    gainMatrices.resize(n,r*maxSimulationSamples);
    gainMatrices.setZero();
    
    errors.resize(r,maxSimulationSamples);
    errors.setZero();
}

void KalmanFilter::predictEstimate(MatrixXd externalInput) {
	estimatesApriori.col(k)=A*estimatesAposteriori.col(k)+B*externalInput;
    covarianceApriori(all,seq(k*n,(k+1)*n-1))=A*covarianceAposteriori(all,seq(k*n,(k+1)*n-1))*(A.transpose())+Q;
    // increment the time step
    k++; 
}

void KalmanFilter::updateEstimate(MatrixXd measurement) {

	// this matrix is used to compute the Kalman gain
	MatrixXd Sk;
    Sk.resize(r,r);
    Sk=R+C*covarianceApriori(all,seq((k-1)*n,k*n-1))*(C.transpose());
    Sk=Sk.inverse();
	// gain matrices 
	gainMatrices(all,seq((k-1)*r,k*r-1))=covarianceApriori(all,seq((k-1)*n,k*n-1))*(C.transpose())*Sk;
    // compute the error - innovation 
	errors.col(k-1)=measurement-C*estimatesApriori.col(k-1);
    // compute the a posteriori estimate, remember that for k=0, the corresponding column is x0 - initial guess
	estimatesAposteriori.col(k)=estimatesApriori.col(k-1)+gainMatrices(all,seq((k-1)*r,k*r-1))*errors.col(k-1);

    MatrixXd In;
    In= MatrixXd::Identity(n,n);
    MatrixXd IminusKC;
    IminusKC.resize(n,n);
    IminusKC=In-gainMatrices(all,seq((k-1)*r,k*r-1))*C;  // I-KC
   
    // update the a posteriori covariance matrix
    covarianceAposteriori(all,seq(k*n,(k+1)*n-1))
    =IminusKC*covarianceApriori(all,seq((k-1)*n,k*n-1))*(IminusKC.transpose())
    +gainMatrices(all,seq((k-1)*r,k*r-1))*R*(gainMatrices(all,seq((k-1)*r,k*r-1)).transpose());
}

MatrixXd KalmanFilter::openData(std::string fileToOpen)
{
	std::vector<double> matrixEntries;
	std::ifstream matrixDataFile(fileToOpen);
	std::string matrixRowString;
	std::string matrixEntry;
	int matrixRowNumber = 0;

	while (getline(matrixDataFile, matrixRowString)) 
	{
		std::stringstream matrixRowStringStream(matrixRowString);

		while (getline(matrixRowStringStream, matrixEntry,','))
		{
			matrixEntries.push_back(stod(matrixEntry)); 
			}
		matrixRowNumber++;
	}

	return Map<Matrix<double, Dynamic, Dynamic, RowMajor>> (matrixEntries.data(), 
															matrixRowNumber, matrixEntries.size() / matrixRowNumber);

}

void KalmanFilter::saveData(std::string estimatesAposterioriFile, std::string estimatesAprioriFile, 
							std::string covarianceAposterioriFile, std::string covarianceAprioriFile, 
							std::string gainMatricesFile, std::string errorsFile) const
{
	const static IOFormat CSVFormat(FullPrecision, DontAlignCols, ", ", "\n");
	
	std::ofstream file1(estimatesAposterioriFile);
	if (file1.is_open())
	{
		file1 << estimatesAposteriori.format(CSVFormat);
		
		file1.close();
	}

	std::ofstream file2(estimatesAprioriFile);
	if (file2.is_open())
	{
		file2 << estimatesApriori.format(CSVFormat);
		file2.close();
	}
	
	std::ofstream file3(covarianceAposterioriFile);
	if (file3.is_open())
	{
		file3 << covarianceAposteriori.format(CSVFormat);
		file3.close();
	}

	std::ofstream file4(covarianceAprioriFile);
	if (file4.is_open())
	{
		file4 << covarianceApriori.format(CSVFormat);
		file4.close();
	}

	std::ofstream file5(gainMatricesFile);
	if (file5.is_open())
	{
		file5 << gainMatrices.format(CSVFormat);
		file5.close();
	}

	std::ofstream file6(errorsFile);
	if (file6.is_open())
	{
		file6 << errors.format(CSVFormat);
		file6.close();
	}
	
}