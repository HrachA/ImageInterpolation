#include "Interpolate.h"

void interpolateNearest(int i, int j, Mat& src, Mat& dst, double scale)
{
	int i1 = i / scale;
	int j1 = j / scale;
	dst.at<Vec3b>(i, j) = src.at<Vec3b>(i1 + 0.5, j1 + 0.5);
}

double interpolateLinear(double p[2], double x)
{
	return p[0] * (1 - x) + p[1] * x;
}

void interpolateBilinear(int i, int j, Mat& src, Mat& dst, double scale)
{
	double i1 = i / scale;
	double j1 = j / scale;
	int y = (int)i1;
	int x = (int)j1;
	int y_1 = y + 1 < src.rows ? y + 1 : y;
	int x_1 = x + 1 < src.cols ? x + 1 : x;
	double deltaY = i1 - y;
	double deltaX = j1 - x;

	for (int k = 0; k < 3; ++k)
	{
		double p[2][2] = {
			{static_cast<double>(src.at<Vec3b>(y, x)[k]),
				static_cast<double>(src.at<Vec3b>(y,x_1)[k])},
			{static_cast<double>(src.at<Vec3b>(y_1, x)[k]),
				static_cast<double>(src.at<Vec3b>(y_1, x_1)[k])},
		};

		double arr[2];
		arr[0] = interpolateLinear(p[0], deltaX);
		arr[1] = interpolateLinear(p[1], deltaX);

		double res = interpolateLinear(arr, deltaY);
		res = res > 255 ? 255 : res;
		res = res < 0 ? 0 : res;
		dst.at<Vec3b>(i, j)[k] = res;
	}
}

double interpolateCubic(double p[4], double x)
{
	return p[1] + 0.5 * x * (p[2] - p[0] + x * (2.0 * p[0] - 5.0*p[1] + 4.0*p[2] - p[3] + x * (3.0 * (p[1] - p[2]) + p[3] - p[0])));
}

void interpolateBicubic(int i, int j, Mat& src, Mat& dst, double scale)
{
	double i1 = i / scale;
	double j1 = j / scale;
	int y = (int)i1;
	int x = (int)j1;
	int y_1 = y + 1 < src.rows ? y + 1 : y;
	int x_1 = x + 1 < src.cols ? x + 1 : x;
	int y_2 = y_1 + 1 < src.rows ? y_1 + 1 : y_1;
	int x_2 = x_1 + 1 < src.cols ? x_1 + 1 : x_1;
	int y_0 = y - 1 >= 0 ? y - 1 : y;
	int x_0 = x - 1 >= 0 ? x - 1 : x;
	double deltaY = i1 - y;
	double deltaX = j1 - x;
	for (int k = 0; k < 3; ++k)
	{
		double p[4][4] = { {static_cast<double>(src.at<Vec3b>(y_0, x_0)[k]),
			static_cast<double>(src.at<Vec3b>(y_0, x)[k]),
			static_cast<double>(src.at<Vec3b>(y_0, x_1)[k]),
			static_cast<double>(src.at<Vec3b>(y_0, x_2)[k])},
			{static_cast<double>(src.at<Vec3b>(y, x_0)[k]),
				static_cast<double>(src.at<Vec3b>(y, x)[k]),
				static_cast<double>(src.at<Vec3b>(y, x_1)[k]),
				static_cast<double>(src.at<Vec3b>(y, x_2)[k])},
			{static_cast<double>(src.at<Vec3b>(y_1, x_0)[k]),
				static_cast<double>(src.at<Vec3b>(y_1, x)[k]),
				static_cast<double>(src.at<Vec3b>(y_1, x_1)[k]),
				static_cast<double>(src.at<Vec3b>(y_1, x_2)[k])},
			{static_cast<double>(src.at<Vec3b>(y_2, x_0)[k]),
				static_cast<double>(src.at<Vec3b>(y_2, x)[k]),
				static_cast<double>(src.at<Vec3b>(y_2, x_1)[k]),
				static_cast<double>(src.at<Vec3b>(y_2, x_2)[k])} };

		double arr[4];
		arr[0] = interpolateCubic(p[0], deltaX);
		arr[1] = interpolateCubic(p[1], deltaX);
		arr[2] = interpolateCubic(p[2], deltaX);
		arr[3] = interpolateCubic(p[3], deltaX);

		double res = interpolateCubic(arr, deltaY);
		res = res > 255 ? 255 : res;
		res = res < 0 ? 0 : res;
		dst.at<Vec3b>(i, j)[k] = res;
	}
}

void resizeNearest(Mat& src, Mat& dst, double scale)
{
	for (int i = 0; i < dst.rows; ++i)
	{
		for (int j = 0; j < dst.cols; ++j)
		{
			interpolateNearest(i, j, src, dst, scale);
		}
	}
}

void resizeLinear(Mat& src, Mat& dst, double scale)
{
	for (int i = 0; i < dst.rows; ++i)
	{
		for (int j = 0; j < dst.cols; ++j)
		{
			interpolateBilinear(i, j, src, dst, scale);
		}
	}
}

void resizeCubic(Mat& src, Mat& dst, double scale)
{
	for (int i = 0; i < dst.rows; ++i)
	{
		for (int j = 0; j < dst.cols; ++j)
		{
			interpolateBicubic(i, j, src, dst, scale);
		}
	}
}

void resizeMixed(Mat& src, Mat& dst, double scale)
{
	Mat gradient;
	Mat greyMat(src.size(), CV_8U);
	cv::cvtColor(src, greyMat, cv::COLOR_BGR2GRAY);

	Sobel(greyMat, gradient, CV_32F, 1, 1, 5);

 //	for (int i = 0; i < gradient.rows; ++i)
	//{
	//	for (int j = 0; j < gradient.cols; ++j)
	//	{
	//		cout << (float)gradient.at<float>(i, j) << " ";
	//	}
	//}

	//imwrite("colorSmallGradient.png", gradient);
	
	//int nn = 0;
	//int bl = 0;
	//int bc = 0;

	for (int i = 0; i < dst.rows; ++i)
	{
		for (int j = 0; j < dst.cols; ++j)
		{
			double i1 = i / scale;
			double j1 = j / scale;
			int y = (int)i1;
			int x = (int)j1;
			int y_1 = y + 1 < src.rows ? y + 1 : y;
			int x_1 = x + 1 < src.cols ? x + 1 : x;
			if (abs(gradient.at<float>(y, x)) < 1 && abs(gradient.at<float>(y_1, x_1)) < 1 &&
				abs(gradient.at<float>(y_1, x)) < 1 && abs(gradient.at<float>(y, x_1)) < 1)
			{
				interpolateNearest(i, j, src, dst, scale);
				//nn++;
			}
			else
			if (abs(gradient.at<float>(y, x)) < 300 && abs(gradient.at<float>(y_1, x_1)) < 300)
			{
				interpolateBilinear(i, j, src, dst, scale);
				//bl++;
			}
			else
			{
				interpolateBicubic(i, j, src, dst, scale);
				//bc++;
			}
		}
	}
	//cout << nn << ' ' << bl << ' ' << bc << endl;
}

Mat Resize(Mat& src, double scale, InterpolationType type)
{
	Mat dst(src.rows * scale, src.cols * scale, src.type(), Scalar(0, 0, 0));
	auto start = chrono::steady_clock::now();

	switch (type)
	{
	case NEAREST:
		resizeNearest(src, dst, scale);
		break;
	case BILINEAR:
		resizeLinear(src, dst, scale);
		break;
	case BICUBIC:
		resizeCubic(src, dst, scale);
		break;
	case MIXED:
		resizeMixed(src, dst, scale);
		break;
	default:
		resizeNearest(src, dst, scale);
		break;
	}

	auto end = chrono::steady_clock::now();
	auto diff = end - start;
	cout << chrono::duration <double, milli>(diff).count() << " ms" << endl;
	return dst;
}

Mat getControlPoints(Mat& src, double scale)
{
	Mat dst(src.rows * scale, src.cols * scale, src.type(), Scalar(0, 0, 0));

	for (int i = 0; i < dst.rows; ++i)
	{
		for (int j = 0; j < dst.cols; ++j)
		{
			double i1 = i / scale;
			double j1 = j / scale;
			if (i1 - (int)i1 == 0 && j1 - (int)j1 == 0)
			{
				dst.at<Vec3b>(i, j) = src.at<Vec3b>(i1, j1);
			}
		}
	}
	return dst;
}