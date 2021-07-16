EIGEN = /opt/local/include/eigen3/
Stoch = /Users/anabel/Documents/PhD/Code/Bayesian/StochTK++
ST_I  = $(STOCH)/includes/

BINDIR = ./

LIBS    = -framework Accelerate -lc++ -lpcb++ -lcub++ -lnlopt 
LIBDIR	= /opt/local/lib  -L$(Stoch)/lib -L/usr/local/lib
INCLUDES= -I$(EIGEN) -I$(Stoch)/includes -I/usr/local/include 
CFLAGS	=  -O3 '-DCOMPLEX=std::complex<double>' 
CPP     = g++
CODE	= mcmc_c++.exe

OBJECT= mcmc_c++.o

code : $(OBJECT)
	$(CPP) -o $(CODE) $(CFLAGS) $(OBJECT) -L$(LIBDIR) $(LIBS)
 
mcmc_c++.o : mcmc_c++.cpp $(INCLS)
	$(CPP)  $(CFLAGS) $(INCLUDES) -c mcmc_c++.cpp 

clean:
	rm -f $(OBJECT) $(OBJECTD) $(OBJECTX) *~  core* $(CODE) fort.*
