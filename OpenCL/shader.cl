__kernel void init(__global int *count, __global int *distance, __global int *flag){
    //int idx = get_global_id(0);
    //count[idx] = 0;
    //distance[idx] = -1;
    //flag[idx] = 0;
}
/*
__kernel void dijkstra(__global float *d_Result, __global float *d_Input, int N){
    const int tid = get_global_id(0);
    const int threadN = get_global_size(0);

    // Block index
    int bx = get_group_id(0);
    int by = get_group_id(1);

    // Thread index
    int tx = get_local_id(0);
    int ty = get_local_id(1);
    
    float sum = 0;
    
    for(int pos = tid; pos < N; pos += threadN)
        sum += d_Input[pos];
    
    d_Result[tid] = sum;
}
*/