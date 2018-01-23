OPTIONS = 
CFILES = 
CUFILES = 
MODE = 
debug = 
OPTIMIZATION = -xAVX 

# get architecture

ARCH = $(shell uname -m)
OLD_ARCH = $(shell cat makefile.log)

#default settings

#OC = gcc
OC = icc
#OC = kinst-ompp icc
OFLAGS = -O3 -fopenmp $(OPTIMIZATION)
LDFLAGS = -lm -I../LinA/include/ 
#-lmpfr -lgmp
LIB = -L../LinA/lib -lLinAlg
BASICFILES = File_stuff.c linear_algebra.c PCGS.c Multigrid.c stack.c misc.c red_black_tree.c reflective_newton.c BiCGStab.c GMRES_Newton.c

ifeq ($(strip $(debug)),yes)
	OFLAGS = -Wall -fopenmp -g -pg
endif

ifeq ($(strip $(MODE)),cuda)

	OC = nvcc
	OFLAGS = -arch=sm_10 --compiler-bindir=/usr/bin/gcc-4.3
	LDFLAGS = -I../LinA/include -I/usr/local/lib64/cuda/include
	LIB = -lm -L../LinA/lib -lLinAlg -L/usr/local/lib64/cuda/lib64 -lcudart -L/usr/local/lib64/cuda/lib -lcusparse -lcublas
	FILETYPE = cu
	CFILES = $(BASICFILES) cudaLinA.c
	CUFILES = $(CFILES:%.c=%.cu)
else 
	FILETYPE = c
	CFILES = $(BASICFILES) gmres.c utils.c
endif



OFILES = $(CFILES:%.c=%.o)

lib: look $(OFILES)
	@echo "========================= Building library ================================="
	ar -r libLinAlg.a $(OFILES)
	if [ ! -d include ]; then mkdir include; fi
	cp *.h include
	if [ ! -d lib ]; then mkdir lib; fi
	mv libLinAlg.a lib
	
look: $(CUFILES)

ifneq ($(strip $(ARCH)), $(strip $(OLD_ARCH)))
	@echo new architecture: $(OLD_ARCH) to $(ARCH)
	make clean
else
	@echo $(ARCH) architecture
endif

%.cu : %.c

	cp $< $@

%.o : %.$(FILETYPE)

	$(OC) $(OFLAGS) $(OPTIONS) -c $< $(LDFLAGS)
	@echo $(ARCH) > makefile.log
	
tags:
	rm LinA.c.tags
	touch LinA.c.tags
	geany -Pg LinA.c.tags *.h
	cp LinA.c.tags ~/.config/geany/tags/

clean:
	rm -rf *.o lib include *.log
	rm -f *.cu
