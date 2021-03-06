#ifndef INTEPOLATE_H
#define INTEPOLATE_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <ctime>

using namespace std;
using namespace cv;

enum InterpolationType
{
	NEAREST,
	BILINEAR,
	BICUBIC,
	MIXED
};

void resizeNearest(Mat& src, Mat& dst, double scale);

void resizeLinear(Mat& src, Mat& dst, double scale);

void resizeCubic(Mat& src, Mat& dst, double scale);

void resizeMixed(Mat& src, Mat& dst, double scale);

Mat Resize(Mat& src, double scale, InterpolationType type = NEAREST);

Mat getControlPoints(Mat& src, double scale);

#endif //INTEPOLATE_H