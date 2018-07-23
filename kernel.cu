
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <string>
#include <math.h>
#include <time.h>

using namespace cv;
using namespace std;

clock_t start_t, end_t;

void RGB2GSCALE_CPU(unsigned char *const r_channel, unsigned char *const b_channel, unsigned char *const g_channel, unsigned char *const grsc_img, const size_t rowCount, const size_t colCount);
cudaError_t grayScale(std::string inFile, std::string outFile);
Mat processImage(string inFile, int *blocks, int* rbg_size, int* grsc_size, const int block_size, int *rows, int *columns);
void report_running_time();

//Kernel used to convert all red, blue, and green channels of an image into grayscale based off the luma transform equation
__global__ void grayscaleKernel(const unsigned char *const r_channel, const unsigned char *const b_channel, const unsigned char *const g_channel, unsigned char *const grsc_img, int rbg_size)
{
	int displacement = threadIdx.x + blockIdx.x * blockDim.x;
	// Check to see if thread # is greater than rbg size
	if (displacement < rbg_size){

	const float channelSum = (.299f * r_channel[displacement]) + (.587f * g_channel[displacement]) + (.114f * b_channel[displacement]);
	grsc_img[displacement] = channelSum;
	}

}

int main(int argc, char** argv)
{
    
	cudaError_t cudaStatus = grayScale(argv[1], "output");

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for grayscale CUDA function
cudaError_t grayScale(std::string inFile, std::string outFile)
{
	Mat h_rbg_img;
	unsigned char* d_grsc_img, *d_r_channel, *d_b_channel, *d_g_channel, *h_grsc_img;
	const int blockSize = 512;
	int blocks = 0;
	int grsc_size = 0;
	int rbg_size = 0;
	int rows = 0;
	int columns = 0;

    cudaError_t cudaStatus;

	// This function outputs a cv:Mat that will be split up into channels
	h_rbg_img = processImage(inFile, &blocks, &rbg_size, &grsc_size, blockSize, &rows, &columns);

	//Grayscale images have a single channel, so it may be represented by an array of unsigned char
	h_grsc_img = new unsigned char[grsc_size];

	//An array of 3 channels: blue, green, red.
	Mat bgr[3];
	//Splits the rbg image into the array specified above
	split(h_rbg_img, bgr);

	//Start timer for cpu execution
	start_t = clock();

	//CPU function to convert an image to grayscale
	RGB2GSCALE_CPU(bgr[2].data, bgr[1].data, bgr[0].data, h_grsc_img, rows, columns);

	//Reports running time of cpu version
	report_running_time();

	//Declare the Mats that will write the grayscale image out
	Mat img(rows, columns, CV_8UC1);
	Mat imgGpu(rows, columns, CV_8UC1);
	img.data = h_grsc_img;

	imwrite("CPU.png", img);

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate buffers for color channels and grsc images in device
    cudaStatus = cudaMalloc((void**)&d_r_channel, rbg_size*sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&d_b_channel, rbg_size * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_g_channel, rbg_size * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

    cudaStatus = cudaMalloc((void**)&d_grsc_img, grsc_size*sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

   cudaStatus = cudaMemcpy(d_r_channel, bgr[2].data, rbg_size*sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(d_g_channel, bgr[1].data, rbg_size * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(d_b_channel, bgr[0].data, rbg_size * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	//Timer for CUDA kernel function
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

    // Launch a kernel on the GPU with one thread for each pixel
	grayscaleKernel<<<blocks, blockSize>>>(d_r_channel, d_g_channel, d_b_channel, d_grsc_img, rbg_size);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	//Report execution time
	printf("************ Total Kernel Running Time = %0.5f ms **********\n", elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "grayscaleKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

	//New grsc image to ensure that the one written out is from the device
	h_grsc_img = new unsigned char[grsc_size];
	cudaStatus = cudaMemcpy(h_grsc_img, d_grsc_img, rbg_size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	imgGpu.data = h_grsc_img;
	//Write out GPU version of the grayscale image
	imwrite("GPU.png", imgGpu);

Error:
    cudaFree(d_r_channel);
	cudaFree(d_b_channel);
	cudaFree(d_g_channel);
    cudaFree(d_grsc_img);
    
    return cudaStatus;
}

//cpu implemenetation
//equation chosen for RGB to greyscale is Out = .299f * Red + .587f * Green + .114f * Blue. AKA luma transform
//or 									red channel 					blue channel	 				green channel				grayscale image				rowCount		columnCount
void RGB2GSCALE_CPU(unsigned char *const r_channel, unsigned char *const b_channel, unsigned char *const g_channel, unsigned char *const grsc_img, const size_t rowCount, const size_t colCount) {

	size_t row, col;
	int displacement;

	//brute force for loop iterating over all pixels in image and performing transformation
	for (row = 0; row<rowCount; ++row) {
		for (col = 0; col < colCount; ++col) {
			//displacement is total columns plus current column times current row
			displacement = row * colCount + col;
			//obtain current working pixel
			//apply formula for the greyscale rank
			const float chanSum = (.299f * r_channel[displacement]) + (.587f * g_channel[displacement]) + (.114f * b_channel[displacement]); //these float values vary depending on what formula you use, we selected the most common
			//update greyscale image with the channel sum value
			grsc_img[displacement] = chanSum;


		}
	}


}

//Reads in an image from file, determines the number of GPU blocks required as well as the size of the image
Mat processImage(string inFile, int *blocks, int* rbg_size, int* grsc_size, const int block_size, int *rows, int *columns) {
	//open cv 
	Mat image;
	//open cv
	image = imread(inFile, CV_LOAD_IMAGE_COLOR);

	*rows = image.rows;
	*columns = image.cols;
	*rbg_size = *rows * *columns;
	*grsc_size = *rows * *columns;

	*blocks = ceil(*grsc_size / block_size);

	return image;

}

//Reports CPU running time
void report_running_time() {
	end_t = clock();
	
	printf("Running time of cpu version: %0.5f ms \n", difftime(end_t, start_t));
}


