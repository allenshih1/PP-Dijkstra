all: dijkstra gen

dijkstra: dijkstra.c
	gcc -fopenmp -o dijkstra dijkstra.c

gen: gen.c
	gcc -o gen gen.c
