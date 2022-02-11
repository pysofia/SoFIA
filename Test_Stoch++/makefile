EIGEN = /opt/local/include/eigen3/
Stoch = /Users/anabel/Documents/PhD/Code/Bayesian/StochTK++
ST_I  = $(STOCH)/includes/

BINDIR = ./

LIBS    = -framework Accelerate -lc++ -lpcb++ -lcub++ -lnlopt 
LIBDIR	= /opt/local/lib  -L$(Stoch)/lib -L/usr/local/lib
INCLUDES= -I$(EIGEN) -I$(Stoch)/includes -I/usr/local/include 
CFLAGS	=  -O3 '-DCOMPLEX=std::complex<double>' 
CPP     = g++
CODE	= GPR_MCMC.exe
INCLS   = sampler.h

OBJECT= GPR_MCMC.o

code : $(OBJECT)
	$(CPP) -o $(CODE) $(CFLAGS) $(OBJECT) -L$(LIBDIR) $(LIBS)
 
GPR_MCMC.o : GPR_MCMC.cpp $(INCLS)
	$(CPP)  $(CFLAGS) $(INCLUDES) -c GPR_MCMC.cpp

clean:
	rm -f $(OBJECT) $(OBJECTD) $(OBJECTX) *~  core* $(CODE) fort.* *.gnu
