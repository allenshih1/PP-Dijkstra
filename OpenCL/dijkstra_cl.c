#include <stdio.h>
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SIZE 3000

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
	cl_context context;
	cl_uint num_devices;
	//cl_platform_id *platforms;
	cl_device_id *devices;
	cl_command_queue queue;
	cl_program program;
	cl_kernel init, dijkstra;
    char *devName;
    char *devVer;
	size_t cb;

	/*
	clGetPlatformIDs(0, 0, &num_devices);
	platforms = (cl_platform_id*) malloc(num_devices * sizeof(cl_platform_id));
	clGetPlatformIDs(num_devices, &platforms[0], NULL);
	*/
	// create a GPU context
	context = clCreateContextFromType(NULL, CL_DEVICE_TYPE_GPU, NULL, NULL, NULL);
	// get a list of devices
    clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &cb);
    devices = (cl_device_id*) malloc(cb); 
    clGetContextInfo(context, CL_CONTEXT_DEVICES, cb, devices, 0);
    // get the number of devices
    clGetContextInfo(context, CL_CONTEXT_NUM_DEVICES, 0, NULL, &cb);
    clGetContextInfo(context, CL_CONTEXT_NUM_DEVICES, cb, &num_devices, 0);

    // show devices info
    for(i = 0; i < num_devices; i++)
    {
        // get device name
        clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 0, NULL, &cb);
        devName = (char*) malloc(cb);
        clGetDeviceInfo(devices[i], CL_DEVICE_NAME, cb, devName, NULL);
        printf("Device: %s\n", devName);
        free(devName);
        // get device supports version
        clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, 0, NULL, &cb);
        devVer = (char*) malloc(cb);
        clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, cb, devVer, NULL);
        printf("        supports %s\n", devVer);
        free(devVer);
    }

    // construct command queue
    queue = clCreateCommandQueue(context, devices[1], 0, NULL);

    // create and compile the program object
    program = load_program(context, "shader.cl");
    if(program == 0)
    {
		perror("Error, can't load or build program");
		//clReleaseMemObject();
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

	// create buffers and copy data
	//cl_mem cl_count = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * MAX_SIZE, &count[0], NULL);
	//cl_mem cl_distance = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * MAX_SIZE, &distance[0], NULL);
	//cl_mem cl_flag = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * MAX_SIZE, &flag[0], NULL);
	cl_mem cl_count = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * MAX_SIZE, &count[0], NULL);
	cl_mem cl_distance = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * MAX_SIZE, &distance[0], NULL);
	cl_mem cl_flag = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * MAX_SIZE, &flag[0], NULL);
	if(cl_count == 0 || cl_distance == 0 || cl_flag == 0) {
		perror("Can't create OpenCL buffer");
		clReleaseMemObject(cl_count);
		clReleaseMemObject(cl_distance);
		clReleaseMemObject(cl_flag);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		return 0;
	}

	scanf("%d", &num_cases);

	for(c = 0; c < num_cases; c++)
	{
		scanf("%d%d", &n, &m);
		///*
		for(i = 0; i < n; i++)
		{
			count[i] = 0;
			distance[i] = -1;
			flag[i] = 0;
		}
		//*/
		

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

		//
		for(i = 0; i < n; i++)
		{
			int min = 1000000000;
			for(j = 0; j < n; j++)
				if(!flag[j] && distance[j] != -1 && distance[j] < min )
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

	// release everything
	clReleaseMemObject(cl_count);
	clReleaseMemObject(cl_distance);
	clReleaseMemObject(cl_flag);
	clReleaseKernel(init);
	//clReleaseKernel(dijkstra);
	clReleaseProgram(program);
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