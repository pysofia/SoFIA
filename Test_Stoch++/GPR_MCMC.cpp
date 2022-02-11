#include <iostream>
#include <iomanip>
#include <fstream>
#include <math.h>
#include <nlopt.hpp>
#include <Eigen/Dense>
#include <random>
#include <functional>
#include "pcb++.h"
#include "cub++.h"
#include "vstoch++.h"
#include "mstoch++.h"
#include "sampler.h"
#include "nelder.h"
#include "gaussian_proc.h"

using namespace std;
using namespace Eigen;

typedef Stoch<PCB,double> DStoch;
typedef VStoch<PCB,double> VStochD;
typedef MStoch<PCB,MatrixXd> MStochD;


/* Evaluate Kernel of the Stochastic process for the two points x and y, given the parameters in par:
	- par(0) is the variance,
	- par(1) is the correlation length
*/

double Kernel(Eigen::VectorXd const &x, Eigen::VectorXd const &y, const Eigen::VectorXd &par){
	double dx = (x-y).norm()/par(1);
	if(par.rows()==3){
		return par(0)*exp( - pow(dx,2)/2. ); /* squared exponential kernel */
	}else if (par.rows() == 4) {
		return par(0)*exp( - pow(dx,par(2))/par(2) ); 
	} else {
		cout << "Which parameterized Kernel ?\n"; exit(1);
	}
};


double myoptfunc_gp(const std::vector<double> &x, std::vector<double> &grad, void *data){
	/* This is the function you optimize for defining the GP */	
	GP* proc = (GP*) data;											//Pointer to the GP
	Eigen::VectorXd p(x.size());									//Parameters to be optimized
	for(int i=0; i<(int) x.size(); i++) p(i) = x[i];				//Setting the proposed value of the parameters
	double value = proc->Set(p);									//Evaluate the function
	if (!grad.empty()) {											//Cannot compute gradient : stop!
		std::cout << "Asking for gradient, I stop !" << std::endl; exit(1);
	}
	neval++;														//increment the number of evaluation count
	return value;
};

const double Big = -1.e16;
DStoch S;
DStoch S_nuis; //ADDED!
GP PS;

double Gqz;

double Gqz_value(double x){
	return pow(10.,-4.0+x*4.);
}
double Gcu_value(double x){
	return pow(10.,-4.0+x*4.); //TPS
}

double GTPS_value(double x){
	return pow(10.,-4.0+x*4.); 
}

double LogPOSTGP(VectorXd const &Xi){
	for(unsigned d=0; d<Xi.rows(); d++){
		if(Xi(d)<0 ) return Big;
		if(Xi(d)>1 ) return Big;
	} 
	return PS.EvalFast(Xi);
};

//double LogPOSTCond(VectorXd const &Xi){
//	VectorXd Xt(3); Xt(0) = Gqz; Xt(1) = Xi(0); Xt(2) = Xi(0)
//	return LogPOST(Xt); 
//};

int main( int argc, char **argv){
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distU(0,1);
	std::normal_distribution<double> distN(0,1);

    int nd			= atoi(argv[1]);
   	unsigned nchain		= (unsigned) atoi(argv[2]);

	FILE* Oin = fopen("/Users/anabel/Documents/PhD/Stagiaires_DCs/Diana/Computations/MTAt1_3D_2/samples.dat","r");
	MatrixXd Xd(3,nd);
	VectorXd Y(nd);

	for(unsigned d=0; d<nd; d++){
		float xx,yy,ww,zz;
		fscanf(Oin,"%f %f %f %f",&xx,&yy,&ww,&zz);
		 Xd(0,d) = xx; Xd(1,d) = yy; Xd(2,d) = ww; Y(d) = zz;//*130;
	}
	fclose(Oin);

	GP proc(Kernel);			//instantiate a Gaussian process
	{
		/*Generate the observations */
		vector<DATA> data;					//We set the training set of the Gaussian process
		for(int i=0; i<nd; i++){
			VectorXd xi = Xd.col(i);
			double f = Y(i);
			DATA dat(xi,f);
			data.push_back(DATA(xi,f));
		}
		proc.SetData(data);					//We provide the training set to the GP
		proc.OptimizeSLE(myoptfunc_gp,4);	//We compute / optimize the GP hyperparameters, put 4 for gamma free
	}

	PS = proc;								//We copy the process to the "global" process PS

	MatrixXd COV = MatrixXd::Zero(nd,nd);
	for(unsigned id=0; id<nd; id++) COV(id,id) = .01; //.01
	FILE* omcmc;

	{	//Sampling the PC approximation
		NelderMeadOptimizer PB(3,LogPOSTGP);
		PB.Initial_Simplex(VectorXd::Ones(3)*.5);
		VectorXd XMAP = PB.OPT();
		printf("\t(canonical) %10.6f %10.6f %10.6f \n",XMAP(0),XMAP(1),XMAP(2));
		printf("\t(gamma val) %10.6f %10.6f %10.6f \n",Gqz_value(XMAP(0)),Gcu_value(XMAP(1)),GTPS_value(XMAP(2)));
		omcmc = fopen("./chain_MTAt3.gnu","w");
		Sample MCMC(COV,LogPOSTGP);							//Instantiate the sampler
		VectorXd XMCMC = XMAP;			//Initialise the chain
    	MCMC.Seed(XMCMC);		
		MCMC.Burn(100000);									//Burn to adapt the covariance
		for(unsigned s=0; s<nchain; s++){
			XMCMC = MCMC.DoStep(1);								//Will record state every 3 steps
			fprintf(omcmc,"%12.6e %12.6e %12.6e \n",Gqz_value(XMCMC(0)),Gcu_value(XMCMC(1)),GTPS_value(XMCMC(2)));
		}
		fclose(omcmc);
		
	}

};