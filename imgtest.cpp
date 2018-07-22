#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <string>

using namespace cv;
using namespace std;

//Mat rgbImage;
//Mat greyImage;

// uchar4	*d_rgbImage;
// unsigned char *d_greyImage 





int main(int argc, char** argv){
	//open cv 
	Mat image;
	Mat im_out;
	//open cv
	image = imread(imageName);
	const int rows = image.rows;
	const int cols = image.cols;
	int elements = rows * cols * 3;
	int size = rows * cols;
	unsigned char *cin = image.data;
	unsigned char *cout = new unsigned char[rows*cols];

	unsigned char* d_in;
	unsigned char* d_out;
	cudaMalloc((void**) &d_in, elements);
	cudaMalloc((void**) &d_out, size);
	cudaMemcpy(d_in, cin, elements*sizeof(unsigned char), cudaMemcpyHostToDevice);
	//Kernal call here
	cudaMemcpy(cout, d_out, size*sizeof(unsigned char), cudaMemcpyDeviceToHost);
	 	

	im_out = Mat(rows, cols, CV8UC1,cout);
	imshow("output.jpg", im_out);
	
	cudaFree(d_in);
	cudaFree(d_out);

}
