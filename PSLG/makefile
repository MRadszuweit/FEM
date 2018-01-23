
CFLAGS = -g -Wall
LDFLAGS = -lm
CFILES = PSLG.c File_stuff.c geometry2D.c
OFILES = $(CFILES:%.c=%.o)

#test: $(CFILES)
#	gcc $(CFLAGS) -o test $(CFILES) $(LDFLAGS)

PSLG: $(OFILES)
	gcc $(CFLAGS) -o PSLG $(OFILES) $(LDFLAGS)

%.o : %.c
	gcc $(CFLAGS) -c $<

clean:
	rm -f *.o PSLG