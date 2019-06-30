
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <fstream>

cudaError_t addWithCuda(unsigned int* p_red, unsigned int* p_green, unsigned int* p_blue, unsigned int size);
int checkSize(char* filename);
void appendHeader(char* filename, char* origin);
void readBMP(char* filename, unsigned int* p_red, unsigned int* p_green, unsigned int* p_blue);
void writeBMP(char* filename, char* origin, unsigned int* p_red, unsigned int* p_green, unsigned int* p_blue);
void get(unsigned int* color_from, unsigned int* color_to, int start, int end);
void set(unsigned int* color_from, unsigned int* color_to, int start, int end);

__global__ void addKernel(unsigned int* p_red, unsigned int* p_green, unsigned int* p_blue)
{
	int i = threadIdx.x;
	int avg = (p_red[i] * 0.2126 + p_green[i] * 0.7152 + p_blue[i] * 0.0722) / 3;
	if (avg > 128)
	{
		avg -= (avg - 128) * 0.5;
	}
	else {
		avg += (128 - avg) * 0.5;
	}
	p_red[i] = avg;
	p_green[i] = avg;
	p_blue[i] = avg;
}

int main()
{
	int size = checkSize("C:\\Users\\mateu\\source\\repos\\cuda\\x64\\Debug\\test.bmp");
	unsigned int* red = new unsigned int[size];
	unsigned int* green = new unsigned int[size];
	unsigned int* blue = new unsigned int[size];
	readBMP("C:\\Users\\mateu\\source\\repos\\cuda\\x64\\Debug\\test.bmp", red, green, blue);
	int current_size = size;
	cudaError_t cudaStatus;
	while (current_size > 0)
	{
		if (current_size >= 900)
		{
			unsigned int* temp_red = new unsigned int[900];
			unsigned int* temp_green = new unsigned int[900];
			unsigned int* temp_blue = new unsigned int[900];
			get(red, temp_red, size - current_size, size - current_size + 900);
			get(green, temp_green, size - current_size, size - current_size + 900);
			get(blue, temp_blue, size - current_size, size - current_size + 900);
			cudaStatus = addWithCuda(temp_red, temp_green, temp_blue, 900);
			set(temp_red, red, size - current_size, size - current_size + 900);
			set(temp_green, green, size - current_size, size - current_size + 900);
			set(temp_blue, blue, size - current_size, size - current_size + 900);
			current_size -= 900;
		} else {
			unsigned int* temp_red = new unsigned int[current_size];
			unsigned int* temp_green = new unsigned int[current_size];
			unsigned int* temp_blue = new unsigned int[current_size];
			get(red, temp_red, size - current_size, size);
			get(green, temp_green, size - current_size, size);
			get(blue, temp_blue, size - current_size, size);
			cudaStatus = addWithCuda(temp_red, temp_green, temp_blue, current_size);
			set(temp_red, red, size - current_size, size);
			set(temp_green, green, size - current_size, size);
			set(temp_blue, blue, size - current_size, size);
			current_size -= current_size;
		}
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addWithCuda failed!");
			return 1;
		}
	}
    
    

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
	writeBMP("C:\\Users\\mateu\\source\\repos\\cuda\\x64\\Debug\\output.bmp", "C:\\Users\\mateu\\source\\repos\\cuda\\x64\\Debug\\test.bmp", red, green, blue);
    return 0;
}

void get(unsigned int* color_from, unsigned int* color_to, int start, int end)
{
	int index = 0;
	for (int i = start; i < end; i++)
	{
		color_to[index] = color_from[i];
		i++;
		index++;
	}
}

void set(unsigned int* color_from, unsigned int* color_to, int start, int end)
{
	int index = start;
	for (int i = 0; i < end - start; i++)
	{
		color_to[index] = color_from[i];
		i++;
		index++;
	}
}

int checkSize(char* filename)
{
	FILE* f = fopen(filename, "rb");
	unsigned char info[54];
	fread(info, sizeof(unsigned char), 54, f); // read the 54-byte header

	// extract image height and width from header
	int width = *(int*)&info[18];
	int height = *(int*)&info[22];
	int size = width * height;
	fclose(f);
	return size;
}

void appendHeader(char* filename, char* origin)
{
	FILE* f = fopen(origin, "rb");
	unsigned char* info = new unsigned char [54];
	fread(info, sizeof(unsigned char), 54, f); // read the 54-byte header
	fclose(f);
	f = fopen(filename, "wb");
	fwrite(info, sizeof(unsigned char), 54, f);
	fclose(f);
}

void readBMP(char* filename, unsigned int* p_red, unsigned int* p_green, unsigned int* p_blue)
{
	int i;
	FILE* f = fopen(filename, "rb");
	unsigned char info[54];
	fread(info, sizeof(unsigned char), 54, f); // read the 54-byte header

	// extract image height and width from header
	int width = *(int*)&info[18];
	int height = *(int*)&info[22];

	int size = 3 * width * height;
	unsigned char* data = new unsigned char[size];
	fread(data, sizeof(unsigned char), size, f); // read the rest of the data at once
	fclose(f);

	int index = 0;

	for (i = 0; i < size; i += 3)
	{
		int red = data[i];
		int green = data[i + 1];
		int blue = data[i + 2];
		p_red[index] = red;
		p_green[index] = green;
		p_blue[index] = blue;
		index++;
	}
}

void writeBMP(char* filename, char* originfile, unsigned int* p_red, unsigned int* p_green, unsigned int* p_blue)
{
	int i;
	appendHeader(filename, originfile);

	FILE* f = fopen(filename, "a+b");

	int size = 3 * checkSize(originfile);
	unsigned char* data = new unsigned char[size];

	int index = 0;

	for (i = 0; i < size; i += 3)
	{
		data[i] = p_red[index];
		data[i + 1] = p_green[index];
		data[i + 2] = p_blue[index];
		index++;
	}

	fwrite(data, sizeof(unsigned char), size, f); // read the rest of the data at once
	fclose(f);
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(unsigned int* p_red, unsigned int* p_green, unsigned int* p_blue, unsigned int size)
{
	unsigned int *dev_a = 0;
	unsigned int *dev_b = 0;
	unsigned int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(unsigned int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(unsigned int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, p_red, size * sizeof(unsigned int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, p_green, size * sizeof(unsigned int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(dev_c, p_blue, size * sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_a, dev_b, dev_c);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

	cudaStatus = cudaMemcpy(p_red, dev_a, size * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(p_green, dev_b, size * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(p_blue, dev_c, size * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_a);
    cudaFree(dev_b);
	cudaFree(dev_c);

    return cudaStatus;
}