
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <string>

#include <stdio.h>

void RGB2GSCALE_CPU(const uchar4* const rbg_img, unsigned char *const grsc_img, const size_t rowCount, const size_t colCount);

__global__ void grayscaleKernel(const uchar4* const rbg_img, unsigned char *const grsc_img)
{
	int displacement = threadIdx.x + blockIdx.x * blockDim.x;

	const uchar4 color = rbg_img[displacement];
	const float channelSum = (.299f * color.x) + (.587f * color.y) + (.114f * color.z);
	grsc_img[displacement] = channelSum;

}

int main()
{
    
	cudaError_t cudaStatus;


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
	uchar4 *d_rbg_img, *h_rbg_img;
	unsigned char *d_grsc_img, *h_grsc_img;
	const dim3 blockSize = (512, 1, 1);
	int *blocks;

    cudaError_t cudaStatus;

	// This function should output a uchar4* from the input picture as well as set the block number from the pixel number
	//h_rbg_img = processImage(std::string inFile, blocks, blocksize);

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate buffers for rbg and grsc images in device
    cudaStatus = cudaMalloc((void**)&d_rbg_img, sizeof(uchar4));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&d_grsc_img, sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(d_rbg_img, h_rbg_img, sizeof(uchar4), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	const dim3 gridSize(*blocks, 1, 1);
    // Launch a kernel on the GPU with one thread for each pixel
	grayscaleKernel<<<blockSize, blockSize>>>(d_rbg_img, d_grsc_img);

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

    
	cudaStatus = cudaMemcpy(h_grsc_img, d_grsc_img, sizeof(unsigned char), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(d_rbg_img);
    cudaFree(d_grsc_img);
    
    return cudaStatus;
}

//cpu implemenetation
//equation chosen for RGB to greyscale is Out = .299f * Red + .587f * Green + .114f * Blue. AKA luma transform
//or 			img1 				img2 				rowCount		columnCount
void RGB2GSCALE_CPU(const uchar4* const rbg_img, unsigned char *const grsc_img, const size_t rowCount, const size_t colCount) {

	size_t row, col;
	
	//brute force for loop iterating over all pixels in image and performing 		calculation
	for (row = 0; row<rowCount; ++row) {
		for (col = 0; col < colCount; ++col) {
			//displacement is total columns plus current column times current row
			int displacement = colCount + col * row;
			//obtain current working pixel
			const uchar4 colo = rbg_img[displacement]; //color can be represented using unsigned 4
			//apply formula for the greyscale rank
			const float chanSum = (.299f * colo.x) + (.587f * colo.y) + (.114f * colo.z); //these float values vary depending on what formula you use, we selected the most common
																			  //update greyscale image with the channel sum value
			grsc_img[displacement] = chanSum;


		}
	}


}

