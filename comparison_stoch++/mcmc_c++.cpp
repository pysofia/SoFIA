#include <iostream>
#include <vector>
#include <random>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <fstream>
#include <nlopt.hpp>

// includes from StochTk++
#include "pcb++.h"
#include "cub++.h"
#include "vstoch++.h"
#include "mstoch++.h"
#include "sampler.h"

using namespace std;
using namespace Eigen;

// This script accounts for the full experiment for nitridation in the paper of Helber et al //

// typedef Stoch<PCB,double> DStoch;

double ishigami (double x, double y, double z,double a, double b){
    return sin(x) + a*pow(sin(y),2) + b*pow(z,4)*sin(x);
}

// Likelihood
double log_lik(VectorXd const &Xi){

    double a = 8.;
    double b = 1.;
    double sigma = 0.2;

    return -pow(abs(ishigami(0,1,2,a,b)-ishigami(0,1,2,Xi(0),Xi(1))),2)/(2*pow(sigma*ishigami(0,1,2,a,b),2))-pow(abs(ishigami(0,2,5,a,b)-ishigami(0,2,5,Xi(0),Xi(1))),2)/(2*pow(sigma*ishigami(0,2,5,a,b),2))-pow(abs(ishigami(1,8,5,a,b)-ishigami(1,8,5,Xi(0),Xi(1))),2)/(2*pow(sigma*ishigami(1,8,5,a,b),2));
}

int main (){

    unsigned Ndim = 2; // Our MCMC is going to sample a 5D space;
	MatrixXd COV = MatrixXd::Zero(Ndim,Ndim);
	for(unsigned id=0; id<Ndim; id++) COV(id,id) = .01;
	FILE* omcmc;
    {
        omcmc = fopen("/Users/anabel/Documents/PhD/Code/SoFIA/comparison_stoch++/chain.gnu","w");
        Sample MCMC(COV,log_lik);							//Instantiate the sampler
		VectorXd XMCMC = VectorXd::Ones(2)*.5;			//Initialise the chain
    	MCMC.Seed(XMCMC);
		MCMC.Burn(100000);							//Burn to adapt the covariance
        
        for(unsigned s=0; s<100000; s++){
			XMCMC = MCMC.DoStep(1);
            fprintf(omcmc,"%12.6e %12.6e\n",XMCMC(0),XMCMC(1));
    }
    fclose(omcmc);
}
}