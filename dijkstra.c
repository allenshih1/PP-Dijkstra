#include<stdio.h>
#include<sys/time.h>
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
	int msec, total;
	struct timeval t1, t2;
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
			adj[u][count[u]][0] = v;
			adj[u][count[u]][1] = w;
			adj[v][count[v]][0] = u;
			adj[v][count[v]][1] = w;
			count[u]++;
			count[v]++;
		}

		gettimeofday(&t1, NULL);
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
