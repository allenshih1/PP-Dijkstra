__kernel void init(__global int *count, __global int *distance, __global int *flag, const int n)
{
	int tid = get_global_id(0);
	//if(tid < n)
	//{
		count[tid] = 0;
		distance[tid] = 1e9;
		flag[tid] = 0;
	//}
}

__kernel
void reduce(__global int *distance,
			__global int *flag,
			//__local int *fetch,
			//__local int *fetch_id,
			__global int *group_min,
			__global int *group_min_id,
			__const int n)
{
	int global_id = get_global_id(0);
	int local_id = get_local_id(0);
	int element;
	int item_min = 1e9;
	int item_min_id = global_id;
	int offset;
	int this, next;
	int this_id, next_id;
	__local int fetch[256];
	__local int fetch_id[256];


	// use all work_item to scan through the distance array
	while(global_id < n)
	{
		element = distance[global_id];
		if(!flag[global_id] && (element < item_min))	// not visited
		{
			item_min = element;
			item_min_id = global_id;
		}
		global_id += get_global_size(0);
	}

	// reduce to find group_min
	fetch_id[local_id] = item_min_id;
	fetch[local_id] = item_min;
	//fetch_id[local_id] = item_min_id;
	barrier(CLK_LOCAL_MEM_FENCE);	// wait for all group_item fetched

	for(offset = get_local_size(0) >> 1; offset > 0; offset >>= 1)
	{
		if(local_id < offset)
		{
			next_id = fetch_id[local_id + offset];
			this_id = fetch_id[local_id];
			next = fetch[local_id + offset];
			this = fetch[local_id];
			if(this < next)
			{
				fetch_id[local_id] = this_id;
				fetch[local_id] = this;
			}
			else
			{
				fetch_id[local_id] = next_id;
				fetch[local_id] = next;
			}
			//fetch[local_id] = (this < next) ? this : next;
		}
		barrier(CLK_LOCAL_MEM_FENCE);	// wait for all group_min are gotten
	}
	if(local_id == 0)
	{
		int gid = get_group_id(0);
		group_min_id[gid] = fetch_id[0];
		group_min[gid] = fetch[0];
	}
/*
  for(int offset = get_local_size(0) / 2;
      offset > 0;
      offset = offset / 2) {
    if (local_id < offset) {
      int other = fetch[local_id + offset];
      int mine = fetch[local_id];
      fetch[local_id] = (mine < other) ? mine : other;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  */
/*
  if (local_id == 0) {
    group_min[get_group_id(0)] = fetch[0];
  }
  */

}

__kernel
void extractMin(__global int *group_min,
				__global int *group_min_id,
				const int n,
				__global int *min,
				__global int *min_id,
				__global int *flag)
{
	int global_id = get_global_id(0);
	int local_id = get_local_id(0);
	int offset;
	int this, next;
	int this_id, next_id;

	for(offset = get_local_size(0) >> 1; offset > 0; offset >>= 1)
	{
		if(local_id < offset)
		{
			next = group_min[local_id + offset];
			next_id = group_min_id[local_id + offset];
			this = group_min[local_id];
			this_id = group_min_id[local_id];
			if(this < next)
			{
				group_min[local_id] = this;
				group_min_id[local_id] = this_id;
			}
			else
			{
				group_min[local_id] = next;
				group_min_id[local_id] = next_id;
			}
			//group_min[local_id] = (this < next) ? this : next;
		}
		barrier(CLK_LOCAL_MEM_FENCE);	// wait for all group_min are gotten
	}
	if(local_id == 0)
	{
		min[get_group_id(0)] = group_min[0];
		min_id[get_group_id(0)] = group_min_id[0];
		flag[min_id[get_group_id(0)]] = 1;
	}
}

__kernel void relax(__global int *adj,
					__global int *distance,
					__global int *weight,
					__global int *count,
					int max_size,
					__global int *min_id)
{
	int k = count[min_id[0]];
	int global_id = get_global_id(0);
	if(global_id < k)
	{
		if( distance[adj[min_id[0]*max_size+global_id]] > distance[min_id[0]] + weight[min_id[0]*max_size+global_id])
		{
			distance[adj[min_id[0]*max_size+global_id]] = distance[min_id[0]] + weight[min_id[0]*max_size+global_id];
		}
	}
}
