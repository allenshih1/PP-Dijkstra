#include <stdio.h>
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SIZE 10000
#define INFINITY 1000000000
#define LOCAL_WORK_SIZE 256

int adj[MAX_SIZE][MAX_SIZE][2];
int count[MAX_SIZE];
int distance[MAX_SIZE];
int flag[MAX_SIZE];

cl_program load_program(cl_context context, const char* filename);

int main(void)
{
	int c, num_cases;
	int n, m, u, v, w;
	int i, j, k;

	// OpenCL
	cl_platform_id *platforms;
	cl_context context;
	cl_uint num_devices;
	cl_device_id *devices;
	cl_command_queue queue;
	cl_program program;
	cl_kernel init, reduce, extractMin, dijkstra;
    char *devName;
    char *devVer;
	size_t cb;
	cl_int err;
	cl_uint num;
	size_t actual_work_size;
	size_t global_work_size;
	size_t local_work_size;
	size_t num_groups;

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
        devName[cb-2] = 0;
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
	//cl_mem cl_count = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * MAX_SIZE, &count[0], NULL);
	//cl_mem cl_distance = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * MAX_SIZE, &distance[0], NULL);
	//cl_mem cl_flag = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * MAX_SIZE, &flag[0], NULL);
	cl_mem cl_count = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * MAX_SIZE, NULL, NULL);
	cl_mem cl_distance = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * MAX_SIZE, NULL, NULL);
	cl_mem cl_flag = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * MAX_SIZE, NULL, NULL);
	if(cl_count == 0 || cl_distance == 0 || cl_flag == 0)
	{
		perror("Can't create OpenCL buffer");
		clReleaseMemObject(cl_count);
		clReleaseMemObject(cl_distance);
		clReleaseMemObject(cl_flag);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}
	//cl_mem cl_min = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, NULL);
	//cl_mem cl_minID = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, NULL);

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
		//clReleaseMemObject();
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}
	clSetKernelArg(init, 0, sizeof(cl_mem), &cl_count);
	clSetKernelArg(init, 1, sizeof(cl_mem), &cl_distance);
	clSetKernelArg(init, 2, sizeof(cl_mem), &cl_flag);
	/*
	reduce = clCreateKernel(program, "reduce", NULL);
	if(reduce == 0)
	{
		perror("Error, can't load kernel reduce");
		clReleaseProgram(program);
		//clReleaseMemObject();
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}
	clSetKernelArg(reduce, 0, sizeof(cl_mem), &cl_distance);
	clSetKernelArg(reduce, 1, sizeof(cl_mem), &cl_flag);
	clSetKernelArg(reduce, 2, sizeof(cl_mem), &cl_min);
	clSetKernelArg(reduce, 3, sizeof(cl_mem), &cl_minID);
	*/
	/*
	dijkstra = clCreateKernel(program, "dijkstra", NULL);
	if(dijkstra == 0)
	{
		perror("Error, can't load kernel dijkstra");
		clReleaseProgram(program);
		//clReleaseMemObject();
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}
	*/

	scanf("%d", &num_cases);

	for(c = 0; c < num_cases; c++)
	{
		scanf("%d%d", &n, &m);
		/*
		for(i = 0; i < n; i++)
		{
			count[i] = 0;
			distance[i] = -1;
			flag[i] = 0;
		}
		*/
		///*
		// invoke kernel init
		actual_work_size = n;
		local_work_size = 256;
		global_work_size = n  + (256 - n % 256);	// must be the multiple of local_work_size
		//size_t num_groups = global_work_size / local_work_size;
		clSetKernelArg(init, 3, sizeof(int), &n);
		err = clEnqueueNDRangeKernel(queue, init, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
		if(err == CL_SUCCESS)
		{
			clEnqueueReadBuffer(queue, cl_count, CL_TRUE, 0, sizeof(int) * actual_work_size, &count[0], 0, NULL, NULL);
			clEnqueueReadBuffer(queue, cl_distance, CL_TRUE, 0, sizeof(int) * actual_work_size, &distance[0], 0, NULL, NULL);
			clEnqueueReadBuffer(queue, cl_flag, CL_TRUE, 0, sizeof(int) * actual_work_size, &flag[0], 0, NULL, NULL);
		}
		//*/

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

		distance[0] = 0;

		// dijkstra
		for(i = 0; i < n; i++)
		{
			int min = INFINITY;
			int minID;
			//clEnqueueWriteBuffer(queue, cl_min, CL_TRUE, 0, sizeof(int), &min, 0, NULL, NULL);
			///*
			for(j = 0; j < n; j++)
				if(!flag[j] && distance[j] != -1 && distance[j] < min )
				{
					min = distance[j];
					k = j;
				}
			//*/
			// invoke kernel reduce
			/*
			//local_work_size = 256;
			//global_work_size = n  + (256 - n % 256);
			num_groups = global_work_size / local_work_size;
			cl_mem cl_group_min = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * num_groups, NULL, NULL);
			clSetKernelArg(reduce, 4, sizeof(cl_mem), &cl_group_min);
			clSetKernelArg(reduce, 5, sizeof(int), &n);
			err = clEnqueueNDRangeKernel(queue, reduce, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
			if(err == CL_SUCCESS)
			{
				clEnqueueReadBuffer(queue, cl_min, CL_TRUE, 0, sizeof(int), &min, 0, NULL, NULL);
				clEnqueueReadBuffer(queue, cl_minID, CL_TRUE, 0, sizeof(int), &minID, 0, NULL, NULL);
			}
			*/

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

	// release kernel
	clReleaseKernel(init);
	//clReleaseKernel(reduce);
	//clReleaseKernel(dijkstra);
	clReleaseProgram(program);
	clReleaseMemObject(cl_count);
	clReleaseMemObject(cl_distance);
	clReleaseMemObject(cl_flag);
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
