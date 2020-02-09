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

#include "Interpolate.h"

void interpolateNearest(int i, int j, Mat& src, Mat& dst, double scale);

double interpolateLinear(double p[2], double x);

void interpolateBilinear(int i, int j, Mat& src, Mat& dst, double scale);

double interpolateCubic(double p[4], double x);

void interpolateBicubic(int i, int j, Mat& src, Mat& dst, double scale);

void resizeNearest(Mat& src, Mat& dst, double scale);

void resizeLinear(Mat& src, Mat& dst, double scale);

void resizeCubic(Mat& src, Mat& dst, double scale);

void resizeMixed(Mat& src, Mat& dst, double scale);

Mat Resize(Mat& src, double scale, InterpolationType type = NEAREST);

Mat getControlPoints(Mat& src, double scale);

#endif //INTEPOLATE_H