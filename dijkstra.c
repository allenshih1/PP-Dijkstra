#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>
#include"omp.h"

#define MAX_SIZE 10000

int adj[MAX_SIZE][MAX_SIZE];
int weight[MAX_SIZE][MAX_SIZE];
int count[MAX_SIZE];
int distance[MAX_SIZE];
int flag[MAX_SIZE];
int nthreads, chunk;

void find_my_min();

int main(int argc, char **argv)
{
	if(argc != 2)
	{
		fprintf(stderr, "Usage: %s <num of threads>\n", argv[0]);
		exit(EXIT_FAILURE);
	}
	int c, numOfCase;
	int n, m, u, v, w;
	int i, j, min, k;
	int msec = 0, total = 0;
	struct timeval t1, t2;
	scanf("%d", &numOfCase);

	for(c = 0; c < numOfCase; c++)
	{
		scanf("%d%d", &n, &m);

		for(i = 0; i < n; i++)
		{
			count[i] = 0;
			distance[i] = 1e9;
			flag[i] = 0;
		}

		for(i = 0; i < m; i++)
		{
			scanf("%d%d%d", &u, &v, &w);
			adj[u][count[u]] = v;
			weight[u][count[u]] = w;
			adj[v][count[v]] = u;
			weight[v][count[v]] = w;
			count[u]++;
			count[v]++;
		}

		gettimeofday(&t1, NULL);
		distance[0] = 0;
		omp_set_num_threads(atoi(argv[1]));
#pragma omp parallel private(i,j)
		{
#pragma omp single
			{
				nthreads = omp_get_num_threads();
				chunk = n/nthreads;
			}
			int id = omp_get_thread_num();
			int startv = id * chunk;
			int endv;
			if(id == nthreads-1)
				endv = n;
			else
				endv = startv + chunk;
			for(i = 0; i < n; i++)
			{
#pragma omp single
				{
					min = 1e9;
				}
				int my_min = 1e9;
				int my_k;
				for(j = startv; j < endv; j++)
					if(!flag[j] && distance[j]<my_min )
					{
						my_min = distance[j];
						my_k = j;
					}
#pragma omp critical
				if(my_min < min)
				{
					min = my_min;
					k = my_k;
				}

#pragma omp barrier
#pragma omp single
				{
					chunk = count[k]/nthreads;
					flag[k] = 1;
				}
				int startvr = id * chunk;
				int endvr;
				if(id == nthreads-1)
					endvr = n;
				else
					endvr = startvr + chunk;
				for(j = startvr; j < endvr && j < count[k]; j++)
					if( distance[adj[k][j]] > distance[k] + weight[k][j])
						distance[adj[k][j]] = distance[k] + weight[k][j];
#pragma omp barrier
			}
		}
		gettimeofday(&t2, NULL);
		msec = (t2.tv_sec-t1.tv_sec)*1000;
		msec += (t2.tv_usec-t1.tv_usec)/1000;
		total += msec;

		int max = 0;
		for(i = 0; i < n; i++)
			if(max < distance[i])
				max = distance[i];

		printf("%d\n", max);
		/*printf("%dms\n", msec);*/

	}
	printf("%d ms\n", total);
	return 0;
}
