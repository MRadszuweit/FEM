CC = icc
debug =
#OPTIMIZATION = -xAVX
#CFLAGS = -Wall -fopenmp $(OPTIMIZATION)
CFLAGS = -O3 -fopenmp -xAVX
LDFLAGS = -I../LinA/include -I/usr/include
LIB = -lm -L../LinA/lib -lLinAlg -L/usr/lib64 -lGL -lGLU -lglut -lpng
CFILES = FEMviewer.c ../FEM2D/FEM2D.c ../FEM2D/geometry2D.c 
OFILES = $(CFILES:%.cpp=%.o)

ifeq ($(strip $(debug)),yes)
	CFLAGS = -Wall -fopenmp -g -pg
endif

compile: LinAlg.a $(OFILES)
	
	$(CC) $(CFLAGS) -o FEMviewer $(OFILES) $(LDFLAGS) $(LIB) 

%.o : %.c

	$(CC) $(CFLAGS) -c $< -o $@ $(LDFLAGS)

	@echo $(ARCH) > makefile.log

LinAlg.a:
	cd ../LinA && make debug="$(debug)"

clean:
	rm -f *.o FEMviewer
