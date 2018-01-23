OFILES = 
CUFILES = 
OPTIONS = -rdynamic
OPTIMIZATION = -xAVX
MODE = 
debug = 
LOCAL = /Home/damage/radszuwe/local

# get architecture

ARCH = $(shell arch)
OLD_ARCH = $(shell cat makefile.log) 

#default settings

#GDB = -g -G
GDB = 
#CC = gcc
CC = icc
#CC = kinst-ompp icc
OC = $(CC)
#CFLAGS = -ffast-math -O3	
CFLAGS = -O3 -fopenmp $(OPTIMIZATION)
#CFLAGS = -Wall -fopenmp -g -pg
UMFPACKINC = -I/usr/include/suitesparse
UMFPACKLIB = -L/usr/lib64 -lblas -llapack -lamd -lcolamd -lcholmod -lumfpack
#GTSINC = -I$(LOCAL)/gts-0.7.6/x86_64/include
#GTSLIB = -L$(LOCAL)/gts-0.7.6/x86_64/lib64 -lgts
# Attention: add lib path for gts to LD_LIBRARY PATH !
#GINC = -I/usr/include/glib-2.0 -I/usr/lib64/glib-2.0/include
#GLIB = -L/usr/lib64 -lglib-2.0
#MUMPSINC = -I/Home/damage/radszuwe/local/MUMPS_4.10.0/include
#MUMPSLIB = -L/Home/damage/radszuwe/local/MUMPS_4.10.0/lib -ldmumps -lmumps_common
#MPIINC = -I/usr/lib64/mpi/gcc/openmpi/include
#MPILIB = -L/usr/lib64/mpi/gcc/openmpi/lib64
#SLEPCINC = -I/Home/damage/radszuwe/local/slepc-3.5.4/include
#SLEPCLIB = -L/Home/damage/radszuwe/local/slepc-3.5.4/arch-darwin-c-debug/lib
LDFLAGS = -lm -I../LinA/include $(UMFPACKINC) $(GTSINC) $(GINC)
LIB = -L../LinA/lib -lLinAlg $(UMFPACKLIB) $(GTSLIB) $(GLIB)
CFILES = crack.c ../FEM2D/FEM2D.c ../FEM2D/geometry2D.c ../FEM2D/FEM.c ../FEM2D/conditionfile.c ../FEM2D/K2_tree.c SuiteSparse_config.c

#PETSC_DIR = /Home/damage/radszuwe/local/petsc-3.5.4

ifeq ($(strip $(debug)),yes)
	CFLAGS = -Wall -fopenmp -g -pg
endif


ifeq ($(strip $(MODE)),cuda)

	OC = nvcc
	CC = c++
	CFLAGS = -m64
	NVCCFLAGS = -arch=sm_10 --compiler-bindir=/usr/bin/gcc-4.3 $(GDB)
	LDFLAGS = -I../LinA/include -I/usr/local/lib64/cuda/include
	LIB = -lm -L../LinA/lib -lLinAlg -L/usr/local/lib64/cuda/lib64 -lcudart -L/usr/local/lib64/cuda/lib -lcusparse -lcublas
	FILETYPE = cu
	CUFILES = $(CFILES:%.c=%.cu)	
else
	NVCCFLAGS = 
	FILETYPE = c
endif

OFLAGS = $(CFLAGS)
OFILES = $(CFILES:%.c=%.o)

look: $(CUFILES)

ifeq ($(strip $(MODE)),cuda)
	@echo mode: $(MODE)	
else
	@echo mode: default
endif

ifneq ($(strip $(ARCH)), $(strip $(OLD_ARCH)))
	@echo new architecture: $(OLD_ARCH) to $(ARCH)
	make clean
else 
	@echo $(ARCH) architecture
endif

	make compile
	
%.cu : %.c

	cp $< $@

compile: LinAlg.a $(OFILES)
	
	$(CC) $(CFLAGS) $(OPTIONS) -o crack$(ARCH) $(OFILES) $(LDFLAGS) $(LIB)

%.o : %.$(FILETYPE)

	$(OC) $(NVCCFLAGS) $(OFLAGS) $(OPTIONS) -c $< -o $@ $(LDFLAGS)

	@echo $(ARCH) > makefile.log

LinAlg.a:
	cd ../LinA && make MODE="$(MODE)" debug="$(debug)"

clean:
	rm -f *.o crack
	rm -f *.cu