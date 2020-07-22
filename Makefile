all: dijkstra gen

dijkstra: dijkstra.c
	gcc -O2 -fopenmp -o dijkstra dijkstra.c

gen: gen.c
	gcc -o gen gen.c

run:
	./gen 10000 | ./dijkstra 4