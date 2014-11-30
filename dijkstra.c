#include<stdio.h>
#include"omp.h"

#define MAX_SIZE 10000

int adj[MAX_SIZE][MAX_SIZE][2];
int count[MAX_SIZE];
int distance[MAX_SIZE];
int flag[MAX_SIZE];
int nthreads, chunk;

void find_my_min();

int main(void)
{
	int c, numOfCase;
	int n, m, u, v, w;
	int i, j, min, k;
	scanf("%d", &numOfCase);

	for(c = 0; c < numOfCase; c++)
	{
		scanf("%d%d", &n, &m);

		for(i = 0; i < n; i++)
		{
			count[i] = 0;
			distance[i] = -1;
			flag[i] = 0;
		}

		for(i = 0; i < m; i++)
		{
			scanf("%d%d%d", &u, &v, &w);
			adj[u-1][count[u-1]][0] = v-1;
			adj[u-1][count[u-1]][1] = w;
			adj[v-1][count[v-1]][0] = u-1;
			adj[v-1][count[v-1]][1] = w;
			count[u-1]++;
			count[v-1]++;
		}

		distance[0] = 0;
		omp_set_num_threads(4);
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
					min = 1000000000;
				}
				int my_min = 1000000000;
				int my_k;
				for(j = startv; j < endv; j++)
					if(!flag[j] && distance[j] != -1 && distance[j]<my_min )
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
					if(distance[adj[k][j][0]] == -1 ||
							distance[adj[k][j][0]] > distance[k] + adj[k][j][1])
						distance[adj[k][j][0]] = distance[k] + adj[k][j][1];
#pragma omp barrier
			}
		}

		int max = 0;
		for(i = 0; i < n; i++)
			if(max < distance[i])
				max = distance[i];

		printf("%d\n", max);

	}
  return 0;
}
