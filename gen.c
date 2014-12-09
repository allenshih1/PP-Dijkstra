#include<stdio.h>

int main(void)
{
	int c = 1;
	int n = 10000;
	int i, j, k;
	printf("%d\n", c);
	for(i = 0; i < c; i++)
	{
		printf("%d %d\n", n, n*(n-1)/2);
		for(j = 0; j < n; j++)
			for(k = 0; k < j; k++)
				printf("%d %d 10\n", j, k);
	}
	
}
