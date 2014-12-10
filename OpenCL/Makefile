CFLAGS =
ifeq ($(CC), gcc)
	LDFLAGS = -lOpenCL
else
	LDFLAGS = -framework OpenCL
endif
SOURCES = dijkstra_cl.c
OBJECTS = $(SOURCES:.c=.o)
EXECUTE = dijkstra_cl

all: $(OBJECTS) $(EXECUTE)

$(EXECUTE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@
$(OBJECTS): $(SOURCES)
	$(CC) $(CFLAGS) $(SOURCES) -c

run:
	../gen|./$(EXECUTE)
clean:
	rm -rf *~ *.o $(EXECUTE)
