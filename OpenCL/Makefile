CC = clang
CFLAGS =
LDFLAGS = -framework OpenCL
SOURCES = dijkstra_cl.c
OBJECTS = $(SOURCES:.c=.o)
EXECUTE = dijkstra_cl

all: $(OBJECTS) $(EXECUTE)

$(EXECUTE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@
$(OBJECTS): $(SOURCES)
	$(CC) $(CFLAGS) $(SOURCES) -c

run:
	./$(EXECUTE) < ../test_data/graph-large.in.txt
clean:
	rm -rf *~ *.o $(EXECUTE)
