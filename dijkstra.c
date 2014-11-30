#include<stdio.h>

#define MAX_SIZE 3000

int adj[MAX_SIZE][MAX_SIZE][2];
int count[MAX_SIZE];
int distance[MAX_SIZE];
int flag[MAX_SIZE];

int main(void)
{
	int c, numOfCase;
	int n, m, u, v, w;
	int i, j, k;
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
		for(i = 0; i < n; i++)
		{
			int min = 1000000000;
			for(j = 0; j < n; j++)
				if(!flag[j] && distance[j] != -1 && distance[j]<min )
				{
					min = distance[j];
					k = j;
				}

			for(j = 0; j < count[k]; j++)
				if(distance[adj[k][j][0]] == -1 ||
						distance[adj[k][j][0]] > distance[k] + adj[k][j][1])
					distance[adj[k][j][0]] = distance[k] + adj[k][j][1];
			flag[k] = 1;
		}

		int max = 0;
		for(i = 0; i < n; i++)
			if(max < distance[i])
				max = distance[i];

		printf("%d\n", max);

	}
  return 0;
}
