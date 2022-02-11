#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
using namespace std;
using namespace Eigen;
typedef double (*pfun_t)(VectorXd const &);
// This class stores known values for vectors. It throws unknown vectors.
class SimpNode {
    public:
        SimpNode() {};
		SimpNode(VectorXd const &X, double value){ 
			this->x = X; this->value = value;
		};
		bool operator<(const SimpNode other) const {
			// cout << "Using SimpNode comparator\n";
			// cout << value << " " << other.F();
			if( value < other.F() ){
				//  cout << "\t true \n";
				 return true;
			}
			// cout << "\t false \n";
			return false;
   	 	};
    	bool operator==(SimpNode other) {
			if (other.value != value) return false;
        	return true;
    	};
		void operator=(SimpNode other){
			x = other.X(); 
			value = other.F();
			// SimpNode snew(other.X(),other.F());
			// return snew;
		};
		VectorXd X() const { return x;	};
		double F() const { return value; };
	private:
		double value;
		VectorXd x;
};

 class NelderMeadOptimizer {
	private:
		pfun_t fopt; 
        int dimension;
        double alpha, gamma, rho, sigma;
        double termination_distance;
		vector<SimpNode> Simplex;
    public:
        NelderMeadOptimizer(int dimension, pfun_t fopt, double termination_distance=1.e-7) {
			// cout << "Create of Nelder Mead optimizer \n";
            this->dimension = dimension;
			this->fopt = fopt;
            this->termination_distance = termination_distance;
            alpha = 1;
            gamma = 2;
            rho = -0.5;
            sigma = 0.5;
        };

        // termination criteria: each pair of vectors in the simplex has to
        // have a distance of at most `termination_distance`
        bool done() {
            if (Simplex.size() < dimension) {
                return false;
            }
            for (int i=0; i<dimension+1; i++) {
                for (int j=i; j<dimension+1; j++) {
                    if (i==j) continue;
                    if ((Simplex[i].X()-Simplex[j].X()).norm() > termination_distance) {
                        return false;
                    }
                }
            }
            return true;
        };

        double Range() {
            VectorXd F(dimension+1);
            for (int i=0; i<dimension+1; i++) {
                F(i) = Simplex[i].F();
            }
            return F.maxCoeff()-F.minCoeff();
        };

		//Initialize
		void Initial_Simplex(VectorXd const &Xini, double size = 0.1){
			// cout << "Initialize simplex \n";
			Simplex.clear();
			for(int i=0; i<dimension+1; i++){
				VectorXd X = Xini; 
                if(i<dimension) X(i) += size;
				double v = fopt(X);
				SimpNode sn(X,v);
				Simplex.push_back(sn);
			}
//			cout << "Sort the simplex\n";
			sort(Simplex.begin(), Simplex.end());
		};

	    // used in `step` to sort the vectors
        bool operator()(const SimpNode& a, const SimpNode& b) {
			cout << "using the define comparison\n";
            return a.F() < b.F();
        }

        double BEST_value() const {
            return Simplex[dimension].F();
        }

        VectorXd OPT() {
            while(!done()) {
                sort(Simplex.begin(), Simplex.end());
                VectorXd cog = VectorXd::Zero(dimension); // center of gravity
                for (int i = 1; i<=dimension; i++) cog += Simplex[i].X();
				cog /= (double) dimension;
                SimpNode best  = Simplex[dimension];
                SimpNode worst = Simplex[0];
                SimpNode second_worst = Simplex[1];
			// reflect
                VectorXd refl = cog + (cog - worst.X())*alpha;
				SimpNode Refl(refl,fopt(refl));
                if (Refl.F() > second_worst.F() && Refl.F() < best.F()) {
                    Simplex[0] = Refl;
                } else if ( Refl.F() > best.F()) {
                    VectorXd expa = cog + (cog - worst.X())*gamma;
					SimpNode Expa(expa,fopt(expa));
                    if (Expa.F() > Refl.F()) {
                        Simplex[0] = Expa;
                    } else {
                        Simplex[0] = Refl;
                    }
                } else {
                	// contract
                    VectorXd contr = cog + (cog - worst.X())*rho;
					SimpNode Contr(contr,fopt(contr));
                    if (Contr.F() > worst.F()){
                        Simplex[0] = Contr;
                    } else {
                        for (int i=0; i<dimension; i++) {
							VectorXd x = best.X() + (Simplex[i].X() - best.X())*sigma;
							SimpNode snew(x,fopt(x));
                            Simplex[i] = snew;
                        }
                	}
                }
            }
             // algorithm is terminating, output: simplex' center of gravity
            VectorXd cog = VectorXd::Zero(dimension);
            for (int i = 0; i<=dimension; i++){
                cog += Simplex[i].X();
            }
            VectorXd optim = cog/(double)(dimension+1); 
            cout << "Nelder-Mead optimum fount at " << optim.transpose() << endl;
            cout << "Range of values " << Range() << endl;
            cout << Simplex[dimension].F() << endl;
            return optim;
        };
};

