OFILES = 
OPTIONS = -rdynamic
OPTIMIZATION = -xAVX
MODE = 
debug = 

#CC = gcc
CC = icc
OC = $(CC)
CFLAGS = -fopenmp -O3	
#CFLAGS = -Wall -fopenmp -g -pg
UMFPACKINC = -I/usr/include/suitesparse
UMFPACKLIB = -L/usr/lib64 -lblas -llapack -lamd -lcolamd -lcholmod -lumfpack

LDFLAGS = -lm -I../LinA/include $(UMFPACKINC) -I../FEM2D -I../../EasyRBT
LIB = -L../LinA/lib -lLinAlg $(UMFPACKLIB)
CFILES = flow.c ../FEM2D/FEM2D.c ../FEM2D/geometry2D.c ../FEM2D/FEM.c SuiteSparse_config.c ../../EasyRBT/redblack.c
#../FEM2D/conditionfile.c ../FEM2D/K2_tree.c

ifeq ($(strip $(debug)),yes)
	CFLAGS = -Wall -fopenmp -g -pg
endif

OFLAGS = $(CFLAGS)
OFILES = $(CFILES:%.c=%.o)

compile: LinAlg.a $(OFILES)
	
	$(CC) $(CFLAGS) $(OPTIONS) -o flow $(OFILES) $(LDFLAGS) $(LIB)

%.o : %.c

	$(OC) $(OFLAGS) $(OPTIONS) -c $< -o $@ $(LDFLAGS)

LinAlg.a:
	cd ../LinA && make MODE="$(MODE)" debug="$(debug)"

clean:
	rm -f *.o flow