CFLAGS= -m64 -I"${MKLROOT}/include" -fopenmp -std=c++17
LDFLAGS= -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lprofiler -lpthread -lm -ldl



csr_spmv: csr_spmv.o Makefile
	g++ csr_spmv.o ${LDFLAGS} -o csr_spmv

csr_spmv.o: csr_spmv.cpp ../conversions.h Makefile
	g++ -c csr_spmv.cpp $(CFLAGS) -o csr_spmv.o
