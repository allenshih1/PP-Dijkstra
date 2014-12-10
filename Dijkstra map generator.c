//
//  Dijkstra map generator.c
//  
//
//  Created by Wellis on 2014/11/30.
//
//

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main (){
    int i,j;
    int numberof_node, maxnode,cluster_in;
    int* node_count;
    int* edge_in_clus;
    printf("Input number of cluster and max nodes in each cluster central point:\n");
    scanf("%d %d",&numberof_node,&maxnode);
    
    node_count=(int*)malloc(numberof_node*sizeof(int));
    edge_in_clus=(int*)malloc(numberof_node*sizeof(int));
    
    int totalnode=0;
    int totaledge=0;
    
    totalnode+=numberof_node;
    totaledge+=numberof_node-1;
    
    srand(time(NULL));
    
    for(i=0;i<numberof_node-1;i++){
        node_count[i]=rand()%maxnode+1;
        totalnode+=node_count[i];
        totaledge+=node_count[i];
        printf("%d have %d node",i,node_count[i]);
        if(node_count[i]>1)edge_in_clus[i]=rand()%(node_count[i]*(node_count[i]-1)/2); //n point at most n(n-1)/2 edge
        else edge_in_clus[i]=0;
        printf(" have %d edge between the nodes\n",edge_in_clus[i]);
        totaledge+=edge_in_clus[i];
    }
    node_count[numberof_node-1]=0;
    edge_in_clus[numberof_node-1]=0;
    
    for(i=0;i<numberof_node;i++){
        printf("%d: %d %d\n",i,node_count[i],edge_in_clus[i]);
    }

    FILE *fp,*fi;
    
    fp=fopen("map.txt","w");
    fprintf(fp,"1\n");
    fprintf(fp,"%d %d\n",totalnode,totaledge);
    
    for(i=0;i<numberof_node-1;i++){
        fprintf(fp,"%d %d %d\n",i,i+1,1000);
    }
    
    int currentstart=numberof_node;
    
    for(i=0;i<numberof_node-1;i++){
        for(j=0;j<node_count[i];j++){
            fprintf(fp,"%d %d %d\n",i,currentstart+j,rand()%1000+1);  ///edge to cluster central
        }
        for(j=0;j<edge_in_clus[i];j++){
            
            int m=rand()%node_count[i]+currentstart;
            int n=rand()%node_count[i]+currentstart;
            if (m==n&&n+1<currentstart+node_count[i])n++;
            else if(n+1>currentstart+node_count[i])n--;
            fprintf(fp,"%d %d %d\n",m,n,rand()%1000+1);  ////edge between point in cluster might repeat
        }
        
        currentstart+=node_count[i];
    }
    fclose(fp);
        ////map =map cat a.txt
}