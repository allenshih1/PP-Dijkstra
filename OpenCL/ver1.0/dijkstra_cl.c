#include <stdio.h>
#include <sys/time.h>
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SIZE 10000
#define LOCAL_WORK_SIZE 256
#define NUM_GROUPS ((MAX_SIZE-(MAX_SIZE%LOCAL_WORK_SIZE))/LOCAL_WORK_SIZE)

int max_size = MAX_SIZE;
int adj[MAX_SIZE][MAX_SIZE];
int weight[MAX_SIZE][MAX_SIZE];
int count[MAX_SIZE];
int distance[MAX_SIZE];
int flag[MAX_SIZE];

cl_program load_program(cl_context context, const char* filename);

int main(void)
{
	int c, num_cases;
	int n, m, u, v, w;
	int i, j, k;
	int msec = 0, total = 0;
	struct timeval t1, t2;

	// OpenCL
	cl_platform_id *platforms;
	cl_context context;
	cl_uint num_devices;
	cl_device_id *devices;
	cl_command_queue queue;
	cl_program program;
	cl_kernel init, reduce, extractMin, relax;
	char *devName;
	char *devVer;
	size_t cb;
	cl_int err;
	cl_uint num;
	size_t actual_work_size;
	size_t global_work_size;
	size_t local_work_size = 256;
	size_t reduce_work_size = 1024;
	size_t num_groups = reduce_work_size/local_work_size;	

	// get the id of supporting OpenCL platforms
	err = clGetPlatformIDs(0, 0, &num);
	if(err != CL_SUCCESS)
	{
		perror("Unable to get platforms");
		return 0;
	}
	platforms = (cl_platform_id*) malloc(num * sizeof(cl_platform_id));
	clGetPlatformIDs(num, &platforms[0], NULL);
	printf("There are %d platform(s) on this device\n", num);
	
	// create a OpenCL context
	cl_context_properties prop[] = { CL_CONTEXT_PLATFORM, (cl_context_properties) platforms[0], 0 };
	context = clCreateContextFromType(prop, CL_DEVICE_TYPE_DEFAULT, NULL, NULL, NULL);
	if(context == 0)
	{
		perror("Can't create OpenCL context");
		return 0;
	}

	// get a list of devices
	clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &cb);
	devices = (cl_device_id*) malloc(cb/sizeof(cl_device_id)); 
	clGetContextInfo(context, CL_CONTEXT_DEVICES, cb, &devices[0], 0);

    // get the number of devices
	clGetContextInfo(context, CL_CONTEXT_NUM_DEVICES, 0, NULL, &cb);
	clGetContextInfo(context, CL_CONTEXT_NUM_DEVICES, cb, &num_devices, 0);
	printf("There are %d device(s) in the context\n", num_devices);

    // show devices info
	for(i = 0; i < num_devices; i++)
	{
		// get device name
		clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 0, NULL, &cb);
		devName = (char*) malloc(sizeof(char) * cb);
		clGetDeviceInfo(devices[i], CL_DEVICE_NAME, cb, &devName[0], NULL);
		devName[cb] = 0;
		printf("Device: %s", devName);
		free(devName);
		
		// get device supports version
		clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, 0, NULL, &cb);
		devVer = (char*) malloc(sizeof(char) * cb);
		clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, cb, &devVer[0], NULL);
		devVer[cb-2] = 0;
		printf(" (supports %s)\n", devVer);
		free(devVer);
	}

    // construct command queue
	queue = clCreateCommandQueue(context, devices[0], 0, NULL);
	if(queue == 0)
	{
		perror("Can't create command queue\n");
		clReleaseContext(context);
		return 0;
	}

	// create buffers and copy data
	cl_mem cl_adj = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * MAX_SIZE * MAX_SIZE, NULL, NULL);
	cl_mem cl_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * MAX_SIZE * MAX_SIZE, NULL, NULL);
	cl_mem cl_count = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * MAX_SIZE, NULL, NULL);
	cl_mem cl_distance = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * MAX_SIZE, NULL, NULL);
	cl_mem cl_flag = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * MAX_SIZE, NULL, NULL);
	cl_mem cl_group_min = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * num_groups, NULL, NULL);
	cl_mem cl_group_min_id = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * num_groups, NULL, NULL);
	cl_mem cl_min = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * 1, NULL, NULL);
	cl_mem cl_min_id = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * 1, NULL, NULL);
	if(cl_adj == 0 || cl_weight == 0 || 
		cl_count == 0 || cl_distance == 0 || cl_flag == 0 ||
		cl_group_min == 0 || cl_group_min_id == 0)
	{
		perror("Can't create OpenCL buffer");
		clReleaseMemObject(cl_count);
		clReleaseMemObject(cl_distance);
		clReleaseMemObject(cl_flag);
		clReleaseCommandQueue(queue);
		clReleaseMemObject(cl_group_min);
		clReleaseMemObject(cl_group_min_id);
		clReleaseContext(context);
		return 0;
	}

    // create and compile the program object
    program = load_program(context, "shader.cl");
    if(program == 0)
    {
		perror("Error, can't load or build program");
		clReleaseMemObject(cl_count);
		clReleaseMemObject(cl_distance);
		clReleaseMemObject(cl_flag);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}

	// create kernel object from compiled program
	init = clCreateKernel(program, "init", NULL);
	if(init == 0)
	{
		perror("Error, can't load kernel init");
		clReleaseProgram(program);
		clReleaseMemObject(cl_count);
		clReleaseMemObject(cl_distance);
		clReleaseMemObject(cl_flag);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}
	clSetKernelArg(init, 0, sizeof(cl_mem), &cl_count);
	clSetKernelArg(init, 1, sizeof(cl_mem), &cl_distance);
	clSetKernelArg(init, 2, sizeof(cl_mem), &cl_flag);
	
	reduce = clCreateKernel(program, "reduce", NULL);
	if(reduce == 0)
	{
		perror("Error, can't load kernel reduce");
		clReleaseProgram(program);
		clReleaseMemObject(cl_count);
		clReleaseMemObject(cl_distance);
		clReleaseMemObject(cl_flag);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}
	clSetKernelArg(reduce, 0, sizeof(cl_mem), &cl_distance);
	clSetKernelArg(reduce, 1, sizeof(cl_mem), &cl_flag);
	clSetKernelArg(reduce, 2, sizeof(cl_mem), &cl_group_min);
	clSetKernelArg(reduce, 3, sizeof(cl_mem), &cl_group_min_id);
	
	extractMin = clCreateKernel(program, "extractMin", NULL);
	if(extractMin == 0)
	{
		perror("Error, can't load kernel extractMin");
		clReleaseProgram(program);
		clReleaseMemObject(cl_count);
		clReleaseMemObject(cl_distance);
		clReleaseMemObject(cl_flag);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}
	clSetKernelArg(extractMin, 0, sizeof(cl_mem), &cl_group_min);
	clSetKernelArg(extractMin, 1, sizeof(cl_mem), &cl_group_min_id);
	clSetKernelArg(extractMin, 2, sizeof(int), &num_groups);
	clSetKernelArg(extractMin, 3, sizeof(cl_mem), &cl_min);
	clSetKernelArg(extractMin, 4, sizeof(cl_mem), &cl_min_id);
	clSetKernelArg(extractMin, 5, sizeof(cl_mem), &cl_flag);

	relax = clCreateKernel(program, "relax", NULL);
	if(relax == 0)
	{
		perror("Error, can't load kernel relax");
		clReleaseProgram(program);
		clReleaseMemObject(cl_count);
		clReleaseMemObject(cl_distance);
		clReleaseMemObject(cl_flag);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}
	clSetKernelArg(relax, 0, sizeof(cl_mem), &cl_adj);
	clSetKernelArg(relax, 1, sizeof(cl_mem), &cl_distance);
	clSetKernelArg(relax, 2, sizeof(cl_mem), &cl_weight);
	clSetKernelArg(relax, 3, sizeof(cl_mem), &cl_count);
	clSetKernelArg(relax, 4, sizeof(int), &max_size);
	clSetKernelArg(relax, 5, sizeof(cl_mem), &cl_min_id);

	scanf("%d", &num_cases);

	for(c = 0; c < num_cases; c++)
	{
		scanf("%d%d", &n, &m);
		///*
		for(i = 0; i < n; i++)
		{
			count[i] = 0;
			distance[i] = 1e9;
			flag[i] = 0;
		}
		//*/
		/*
		// invoke kernel init
		actual_work_size = n;
		local_work_size = 256;
		global_work_size = n  + (256 - n % 256);	// must be the multiple of local_work_size
		//size_t num_groups = global_work_size / local_work_size;
		clSetKernelArg(init, 3, sizeof(int), &n);
		err = clEnqueueNDRangeKernel(queue, init, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
		if(err == CL_SUCCESS)
		{
			cl_event initEvt;		
			clEnqueueReadBuffer(queue, cl_count, CL_TRUE, 0, sizeof(int) * actual_work_size, &count[0], 0, NULL, NULL);
			clEnqueueReadBuffer(queue, cl_distance, CL_TRUE, 0, sizeof(int) * actual_work_size, &distance[0], 0, NULL, NULL);
			//clEnqueueReadBuffer(queue, cl_flag, CL_TRUE, 0, sizeof(int) * actual_work_size, &flag[0], 0, NULL, &initEvt);
			clWaitForEvents(1, &initEvt);
		}
		
		*/

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

		distance[0] = 0;
		cl_event startEvt;
		clEnqueueWriteBuffer(queue, cl_adj, CL_TRUE, 0, sizeof(int) * MAX_SIZE * MAX_SIZE, &adj[0], 0, NULL, NULL);
		clEnqueueWriteBuffer(queue, cl_weight, CL_TRUE, 0, sizeof(int) * MAX_SIZE * MAX_SIZE, &weight[0], 0, NULL, NULL);
		clEnqueueWriteBuffer(queue, cl_count, CL_TRUE, 0, sizeof(int) * MAX_SIZE, &count[0], 0, NULL, NULL);
		clEnqueueWriteBuffer(queue, cl_distance, CL_TRUE, 0, sizeof(int) * MAX_SIZE, &distance[0], 0, NULL, NULL);
		clEnqueueWriteBuffer(queue, cl_flag, CL_TRUE, 0, sizeof(int) * MAX_SIZE, &flag[0], 0, NULL, &startEvt);
		clWaitForEvents(1, &startEvt);
		
		//cl_event setSourceEvt;
		//clEnqueueWriteBuffer(queue, cl_distance, CL_TRUE, 0, sizeof(int) * MAX_SIZE, &distance[0], 0, NULL, &setSourceEvt);
		//clWaitForEvents(1, &setSourceEvt);

		// dijkstra
		gettimeofday(&t1, NULL);
		for(i = 0; i < n; i++)
		{
			int min = 1e9;	// INF
			int min_id;

			///*
			// invoke kernel reduce
			local_work_size = 256;
			global_work_size = reduce_work_size;
			//num_groups = global_work_size / local_work_size;
			clSetKernelArg(reduce, 4, sizeof(int), &n);
			err = clEnqueueNDRangeKernel(queue, reduce, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
			if(err == CL_SUCCESS){}
			else
			{
				printf("Error: can't enqueue kernel reduce\n");
			}
			//*/

			// invoke kernel extractMin
			local_work_size = num_groups;
			global_work_size = num_groups;
			err = clEnqueueNDRangeKernel(queue, extractMin, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
			if(err == CL_SUCCESS)
			{
				//cl_event minEvt;
				//clEnqueueReadBuffer(queue, cl_min, CL_TRUE, 0, sizeof(int) * 1, &min, 0, NULL, NULL);
				//clEnqueueReadBuffer(queue, cl_min_id, CL_TRUE, 0, sizeof(int) * 1, &k, 0, NULL, &minEvt);
				//clWaitForEvents(1, &minEvt);
			}
			else
			{
				printf("Error: can't enqueue kernel extractMin\n");
				printf("%d\n", err);
			}

			//flag[k] = 1;
			//clEnqueueWriteBuffer(queue, cl_flag, CL_TRUE, 0, sizeof(int) * MAX_SIZE, &flag[0], 0, NULL, NULL);

			/*
			cl_event fEvt;
			clEnqueueReadBuffer(queue, cl_count, CL_TRUE, 0, sizeof(int) * MAX_SIZE, &count[0], 0, NULL, NULL);
			clEnqueueReadBuffer(queue, cl_min_id, CL_TRUE, 0, sizeof(int) * 1, &k, 0, NULL, NULL);
			clEnqueueReadBuffer(queue, cl_distance, CL_TRUE, 0, sizeof(int) * MAX_SIZE, &distance[0], 0, NULL, &fEvt);
			clWaitForEvents(1, &fEvt);
			int length = count[k];
			for(j = 0; j < count[k]; j++)
				if( distance[adj[k][j]] > distance[k] + weight[k][j])
				{
					distance[adj[k][j]] = distance[k] + weight[k][j];
				}
			clEnqueueWriteBuffer(queue, cl_distance, CL_TRUE, 0, sizeof(int) * MAX_SIZE, &distance[0], 0, NULL, NULL);
			*/
			

			// invoke kernel relax
			///*
			global_work_size = n + (256 - n % 256);
			local_work_size = 256;
			err = clEnqueueNDRangeKernel(queue, relax, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
			if(err == CL_SUCCESS){}
			else
			{
				printf("Error: can't enqueue kernel relax\n");
			}
			//*/
/*
			int *d = (int*)malloc(sizeof(int) * MAX_SIZE);
			clEnqueueReadBuffer(queue, cl_distance, CL_TRUE, 0, sizeof(int) * MAX_SIZE, &d[0], 0, NULL, NULL);
			int t;
			for(t = 0; t < n; t++) {
				printf("distance[%d] = %d\n", t, d[t]);
			}
			free(d);
*/			
		}
		cl_event finalEvt;
		clEnqueueReadBuffer(queue, cl_distance, CL_TRUE, 0, sizeof(int) * MAX_SIZE, &distance[0], 0, NULL, &finalEvt);
		clWaitForEvents(1, &finalEvt);

		gettimeofday(&t2, NULL);
		msec = (t2.tv_sec-t1.tv_sec)*1000;
		msec += (t2.tv_usec-t1.tv_usec)/1000;
		total += msec;

		int max = 0;
		for(i = 0; i < n; i++)
			if(max < distance[i])
				max = distance[i];

		printf("%d\n", max);
	}

	printf("%d ms\n", total);

	// release kernel
	clReleaseKernel(init);
	clReleaseKernel(reduce);
	clReleaseKernel(extractMin);
	clReleaseKernel(relax);
	clReleaseProgram(program);
	clReleaseMemObject(cl_count);
	clReleaseMemObject(cl_distance);
	clReleaseMemObject(cl_flag);
	clReleaseMemObject(cl_group_min);
	clReleaseMemObject(cl_group_min_id);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	return 0;
}

cl_program load_program(cl_context context, const char* filename)
{
	FILE *fp;
	size_t length;	// long
	char *data;
	const char* source;
	size_t ret;

	// open file
	fp = fopen(filename, "rb");
	if(fp == NULL)
		perror("Error opening file\n");

	// get file length
	fseek (fp, 0, SEEK_END);
	length = ftell (fp);
	fseek (fp, 0, SEEK_SET);	// rewind(fp);

	// read program source
	data = (char*)malloc((length+1) * sizeof(char));
	ret = fread(data, sizeof(char), length, fp);
	if(ret != length)
		perror("Error reading file");
	data[length] = 0;

	// create and build program object
	source = &data[0];
	cl_program program = clCreateProgramWithSource(context, 1, &source, NULL, NULL);
	if(program == 0)
		return 0;

	// compile program
	if(clBuildProgram(program, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS)
		return 0;

	free(data);
	fclose(fp);

	return program;
}
