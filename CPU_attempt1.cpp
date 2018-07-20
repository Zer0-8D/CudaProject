#include <iostream>
#include <string>
#include <stdio.h>


//cpu implemenetation
//equation chosen for RGB to greyscale is Out = .299f * Red + .587f * Green + .114f * Blue. AKA luma transform
//or 			img1 				img2 				rowCount		columnCount
void RGB2GSCALE_CPU( const uchar4* const rbg_img, unsigned char *const grsc_img, const size_t rowCount, const size_t colCount){


	size_t row, col;
	const uchar4 colo;	//color can be represented using unsigned 4
	const float chanSum
	//brute force for loop iterating over all pixels in image and performing 		calculation
	for(row = 0; row<rowCount; ++row){
		for(col < colCount; ++col){
			//displacement is total columns plus current column times current row
			displacement = colCount + col * row;
			//obtain current working pixel
			colo = rbig_img[displacement];
			//apply formula for the greyscale rank
			chanSum = (.299f * colo.x) + (.587f * colo.y) + (.114f * colo.z); //these float values vary depending on what formula you use, we selected the most common
			//update greyscale image with the channel sum value
			grsc_img[displacement] = chanSum; 		


		}
	}



}
