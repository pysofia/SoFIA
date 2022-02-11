
//double myoptfunc_gp(const std::vector<double> &x, std::vector<double> &grad, void *data);


// 	/* This is the function you optimize for defining the GP */	
// double myoptfunc_gp(const std::vector<double> &x, std::vector<double> &grad, void *data){
// 	GP* proc = (GP*) data;											//Pointer to the GP
// 	Eigen::VectorXd p(x.size());									//Parameters to be optimized
// 	for(int i=0; i<(int) x.size(); i++) p(i) = x[i];				//Setting the proposed value of the parameters
// 	double value = proc->Set(p);									//Evaluate the function
// 	if (!grad.empty()) {											//Cannot compute gradient : stop!
// 		std::cout << "Asking for gradient, I stop !" << std::endl; exit(1);
// 	}
// 	neval++;														//increment the number of evaluation count
// 	return value;
// };
//
// double Kernel(Eigen::VectorXd const &x, Eigen::VectorXd const &y, const Eigen::VectorXd &par){
// 	return par(0)*exp( -((x-y)/par(1)).squaredNorm()*.5 ); /* squared exponential kernel */
// };
//
//	Typical procedure:
//		vector<Data> data; 
//	You set your data (couples of points coordinate and function value)
//		GP proc(Kernel);
//		proc.SetData(data);
//		proc.OptimizeSLE();
//		proc.Select();
//

int neval=0;

/* Purpose : select the reduced set of columns of A minimizing the Frobenius error*/

std::vector<int> ColSelect(const Eigen::MatrixXd &A, double frac){
	std::default_random_engine Rng;
	std::uniform_real_distribution<double> Unif(0,1);
	int ns = A.cols();
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
	Eigen::MatrixXd V = svd.matrixV();
	Eigen::VectorXd s = svd.singularValues();
	double st = s.sum();
	double s0 = 0;
	int nr = 0;
	while(s0<st*sqrt(frac)){
		s0 += s(nr); nr++;
		if(nr>=A.cols()) break; 
	}
	nr = fmin(nr*6,A.cols());
	std::cout << s.transpose() << std::endl;
	std::cout<< "Need to make selection of " << nr << " columns over " << ns << "\n";

	Eigen::VectorXd Pr(ns);
	Eigen::VectorXd Prc = Eigen::VectorXd::Zero(ns);
	for(int is=0; is<ns; is++){
		Pr(is) = V.row(is).squaredNorm() / (double)(nr);
		Prc(is) += Pr(is);
		if(is<ns-1) Prc(is+1) = Prc(is);
	}
	Eigen::VectorXi Drawn = Eigen::VectorXi::Zero(ns);
	std::vector<int> draw;
	while(draw.size()<nr){
		double xp = Prc(ns-1)*Unif(Rng);
		int is = 0;
		while(Prc(is)<xp) is++;
		if(Drawn(is)==0){
			draw.push_back(is);
			Drawn(is) = 1;
		}
	}
	Eigen::MatrixXd C(A.rows(),nr);
	for(int ik=0; ik<nr; ik++) C.col(ik) = A.col(draw[ik]);
	Eigen::JacobiSVD<Eigen::MatrixXd> SVD(C, Eigen::ComputeThinU | Eigen::ComputeThinV);
	std::cout << SVD.singularValues().transpose() << std::endl;
	return draw;
};

/* The class for the observations */
class DATA {
	friend class GP;

public:
	DATA(){};
	DATA(Eigen::VectorXd const &x, double const &f){ X=x; F=f;};
	DATA(DATA const &d){ X = d.X; F= d.F;};
	void operator = (const DATA d){ X = d.X; F= d.F;};
	Eigen::VectorXd GetX() const { return X; };
	double GetValue() const { return F; };
	void SetX(Eigen::VectorXd x) { X=x;};
	void SetValue(double f) { F=f;};
private:
	Eigen::VectorXd X;
	double F;
};

/* GP model class */

class GP {

public:
	GP(){ KERNEL = NULL;};
	GP(double (*K)(Eigen::VectorXd const & , Eigen::VectorXd const & ,  const Eigen::VectorXd &)){ KERNEL = K;};

	void SetKernel(double (*K)(Eigen::VectorXd const & , Eigen::VectorXd const & , const Eigen::VectorXd &)){
			KERNEL = K;
	};

	void SetData(const Eigen::VectorXd &X, double F){
			Xpt.push_back(X);
			value.push_back(F);
			nd = Xpt.size();
			std::cout << "The GP will use " << nd << " observations\n";
	};

	void SetData(const std::vector<DATA> &data){
		for(int i=0; i<data.size(); i++){
			Xpt.push_back(data[i].X);
			value.push_back(data[i].F);
		}
		nd = Xpt.size();
		std::cout << "The GP will use " << nd << " observations\n";
	};

	/*Set the Gaussian process for the parameters in par */
	double Set(const Eigen::VectorXd &par){	
		nd = Xpt.size();						//Number of data points
		unsigned np = par.rows();
		Eigen::MatrixXd A(nd,nd);				//Correlation perator
		Eigen::VectorXd Y(nd);					//Observations
		sigsn = pow(par(np-1),2);		//Noise variance
		for(int i=0; i<nd; i++){
			for(int j=i; j<nd; j++){
				A(i,j) = KERNEL(Xpt[i], Xpt[j], par);	//Two points correlation
				if(i!=j){
					A(j,i) = A(i,j);
				}else{
					A(i,j) += sigsn;					//Noise correlation
				}
			}
			Y(i) = value[i];							//Noisy observation
		}
		ldlt.compute(A);  					/* Decompose Correlation */
		Alpha = ldlt.solve(Y);				/* Solve for the GP coordinates*/
		/* Compute log of SLE optimization */
		logp = -Y.dot(Alpha)*.5 - (ldlt.vectorD().array().log()).sum() 
		- (double)(nd)*log(M_PI*2)*.5;	
		return -logp;
	};

	/* Evaluate GP at point x, mean prediction only */
	double EvalFast(Eigen::VectorXd const &x) const {
		double val = 0;
		for(int i=0; i<nd; i++){
			val += KERNEL(x,Xpt[i],PAR)*Alpha(i);
		}
		return val;
	};

	/* Evaluate GP at point x, mean and variance of prediction*/
	Eigen::VectorXd Eval(Eigen::VectorXd const &x) const {
		Eigen::VectorXd kstar(nd);
		for(int i=0; i<nd; i++) kstar(i) = KERNEL(x,Xpt[i],PAR);
		Eigen::VectorXd Out(2);
		Out(0) = kstar.dot(Alpha);
		Eigen::VectorXd v = ldlt.solve(kstar);
		Out(1) = KERNEL(x,x,PAR) - kstar.dot(v);
		return Out;
	};

	double LOGP() const { return logp;};

	void OptimizeSLE(nlopt::vfunc myoptfunc_gp, unsigned const np = 4){
		nlopt::opt opt(nlopt::LN_SBPLX, np); /* algorithm and dimensionality */
		std::vector<double> lb(np); 
		std::vector<double> ub(np);
		std::vector<double> x(np); 
		std::cout << "Optimize Gaussian process for " << np << " hyperparameters\n";
		if(np==3){
			lb[0] = 1.e-4; lb[1] = 1.e-3; lb[2] = 1.e-5;/* lower bounds */
 			ub[0] = 1.e6; ub[1] = 1.e1; ub[2] = 5.; /* upper bounds */
			x[0] = 1; x[1] = 1;  x[2] = 1.;
		}else if(np==4){
			lb[0] = 1.e-4; lb[1] = 1.e-3; lb[2] = 1.1; lb[3] = 1.e-5;/* lower bounds */
 			ub[0] = 1.e6; ub[1] = 1.e1; ub[2] = 2.; ub[3] = 5.; /* upper bounds */
			x[0] = 1; x[1] = 1; x[2] = 1.5; x[3] = 1.;			
		} else {
			std::cout<< "Invalid number of parameters \n";
		}
		opt.set_lower_bounds(lb);
		opt.set_upper_bounds(ub);
		opt.set_min_objective(myoptfunc_gp, this);
		opt.set_xtol_rel(1e-4);
		neval = 0;
		std::cout << "Starting optimization\n";
		double minf; /* the minimum objective value, upon return */
		if (opt.optimize(x, minf) < 0) {
			printf("nlopt failed!\n");
		}
		else {
			if(np==3){
				printf("found minimum at L = %8.6g\n \t\t Std = %8.7g \n \t\t Noise = %8.6g\n" , x[0], x[1], x[2]);
			}else{
				printf("found minimum at L = %8.6g\n \t\t Std = %8.7g \n \t\t Gamma= %6.5g \n\t\t Noise = %8.6g\n" , x[0], x[1], x[2], x[3]);				
			}
		}
		std::cout<<"Number of function evaluations " << neval << std::endl;
		PAR = Eigen::VectorXd(np);
		for(unsigned p=0; p<np; p++) PAR(p) =x[p];
		Set(PAR);
	};

	double EvalSelFast(Eigen::VectorXd const &x) const {
		double val = 0;
		for(int i=0; i<Sel.size(); i++){
			val += KERNEL(x,Xpt[Sel[i]],PAR)*BS(i);
		}
		return val;
	};

	Eigen::VectorXd EvalSel(Eigen::VectorXd const &x) const {
		Eigen::VectorXd kstar(Sel.size());
		for(int i=0; i<Sel.size(); i++){
			kstar(i) = KERNEL(x,Xpt[Sel[i]],PAR);
		}
		Eigen::VectorXd Out(2);
		Out(0) = kstar.dot(BS);
		Out(1) = KERNEL(x,x,PAR) - kstar.dot(Rinv*kstar);
		return Out;
	};

	void Select(double prec=.999){
        if(prec>1) prec = .9999;
        if(prec<0) prec = .1;
		Eigen::MatrixXd K(nd,nd);
		Eigen::VectorXd Y(nd);
		for(int i=0; i<nd; i++){
			for(int j=i; j<nd; j++){
				K(i,j) = KERNEL(Xpt[i],Xpt[j],PAR);
				K(j,i) = K(i,j);
			}
			Y(i) = value[i];
		}
		Sel.clear();
		Sel = ColSelect(K, prec); //This is the reduced set of features.
		Eigen::MatrixXd Z(nd,Sel.size());
		for(int i=0; i<Sel.size(); i++) Z.col(i) = K.col(Sel[i]);
		Eigen::MatrixXd R = Z.transpose()*Z/(sigsn) + (Eigen::MatrixXd) Eigen::VectorXd::Ones(Z.cols()).asDiagonal();
		Rinv = R.inverse();
		BS = Rinv*Z.transpose()*Y/sigsn;
		std::cout<< BS.transpose() << std::endl;
	};

private:
	int nd;
	double sigsn;
	double logp;
	Eigen::LDLT<Eigen::MatrixXd> ldlt;
	Eigen::VectorXd Alpha;
	std::vector<Eigen::VectorXd> Xpt;
	std::vector<double> value;
	double (*KERNEL)(const Eigen::VectorXd &, const Eigen::VectorXd &, const Eigen::VectorXd &);
	Eigen::VectorXd PAR;
	std::vector<int> Sel;
	Eigen::VectorXd BS;
	Eigen::MatrixXd Rinv;
};
