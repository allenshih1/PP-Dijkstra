all: dijkstra gen

dijkstra:
	gcc -fopenmp -o dijkstra dijkstra.c

gen:
	gcc -o gen gen.c
