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

void expandImage(Mat& img, Mat& exp)
{
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			exp.at<Vec3b>(i + 1, j + 1) = img.at<Vec3b>(i, j);
		}
	}
	for (int i = 0; i < img.rows; i++)
	{
		exp.at<Vec3b>(i + 1, 0) = img.at<Vec3b>(i, 0);
		exp.at<Vec3b>(i + 1, exp.cols - 1) = img.at<Vec3b>(i, img.cols - 1);
	}
	for (int j = 0; j < img.cols; j++)
	{
		exp.at<Vec3b>(0, j + 1) = img.at<Vec3b>(0, j);
		exp.at<Vec3b>(exp.rows - 1, j + 1) = img.at<Vec3b>(img.rows - 1, j);
	}
	exp.at<Vec3b>(0, 0) = img.at<Vec3b>(0, 0);
	exp.at<Vec3b>(0, exp.cols - 1) = img.at<Vec3b>(0, img.cols - 1);
	exp.at<Vec3b>(exp.rows - 1, 0) = img.at<Vec3b>(img.rows - 1, 0);
	exp.at<Vec3b>(exp.rows - 1, exp.cols - 1) = img.at<Vec3b>(img.rows - 1, img.cols - 1);
	return;
}

double B_0_2(double t)
{
	if (0 <= t && t < 1)
		return t;
	if (1 <= t && t < 2)
		return 2 - t;
	return 0;
}

double B_0_4(double t)
{
	if (0 <= t && t < 1)
		return (t * t * t) / 6;
	if (1 <= t && t < 2)
		return ((-3) * t * t * t + 12 * t * t - 12 * t + 4) / 6;
	if (2 <= t && t < 3)
		return (3 * t * t * t - 24 * t * t + 60 * t - 44) / 6;
	if (3 <= t && t < 4)
		return ((4 - t) * (4 - t) * (4 - t)) / 6;

	return 0;
}

void interpolateNearest(int i, int j, Mat& src, Mat& dst, double scale)
{
	int i1 = i / scale;
	int j1 = j / scale;
	dst.at<Vec3b>(i, j) = src.at<Vec3b>(i1, j1);
}

void interpolateLinear(int i, int j, Mat& src, Mat& dst, double scale)
{
	double i1 = i / scale;
	double j1 = j / scale;
	int y = (int)i1;
	int x = (int)j1;
	int y_1 = y + 1 < src.rows ? y + 1 : y;
	int x_1 = x + 1 < src.cols ? x + 1 : x;
	double deltaY = i1 - y;
	double deltaX = j1 - x;

	auto f_up = src.at<Vec3b>(y, x) * B_0_2(deltaX + 1) + src.at<Vec3b>(y, x_1) * B_0_2(deltaX);
	auto f_down = src.at<Vec3b>(y_1, x) * B_0_2(deltaX + 1) + src.at<Vec3b>(y_1, x_1) * B_0_2(deltaX);

	auto res = f_up * B_0_2(deltaY + 1) + f_down * B_0_2(deltaY);
	dst.at<Vec3b>(i, j) = res;
}

void interpolateCubic(int i, int j, Mat& src, Mat& dst, double scale)
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

	//if (deltaX == 0 && deltaY == 0)
	//	return;

	auto f_y_minus_1 = src.at<Vec3b>(y_0, x_0) * B_0_4(deltaX + 3) +
						src.at<Vec3b>(y_0, x) * B_0_4(deltaX + 2) +
						src.at<Vec3b>(y_0, x_1) * B_0_4(deltaX + 1) +
						src.at<Vec3b>(y_0, x_2) * B_0_4(deltaX);

	auto f_y = src.at<Vec3b>(y, x_0) * B_0_4(deltaX + 3) +
						src.at<Vec3b>(y, x) * B_0_4(deltaX + 2) +
						src.at<Vec3b>(y, x_1) * B_0_4(deltaX + 1) +
						src.at<Vec3b>(y, x_2) * B_0_4(deltaX);

	auto f_y_plus_1 = src.at<Vec3b>(y_1, x_0) * B_0_4(deltaX + 3) +
						src.at<Vec3b>(y_1, x) * B_0_4(deltaX + 2) +
						src.at<Vec3b>(y_1, x_1) * B_0_4(deltaX + 1) +
						src.at<Vec3b>(y_1, x_2) * B_0_4(deltaX);

	auto f_y_plus_2 = src.at<Vec3b>(y_2, x_0) * B_0_4(deltaX + 3) +
						src.at<Vec3b>(y_2, x) * B_0_4(deltaX + 2) +
						src.at<Vec3b>(y_2, x_1) * B_0_4(deltaX + 1) +
						src.at<Vec3b>(y_2, x_2) * B_0_4(deltaX);


	auto res = f_y_minus_1 * B_0_4(deltaY + 3) +
						f_y * B_0_4(deltaY + 2) +
						f_y_plus_1 * B_0_4(deltaY + 1) +
						f_y_plus_2 * B_0_4(deltaY);

	dst.at<Vec3b>(i, j) = res;
}

double cubicInterpolate(double p[4], double x) 
{
	return p[1] + 0.5 * x*(p[2] - p[0] + x * (2.0*p[0] - 5.0*p[1] + 4.0*p[2] - p[3] + x * (3.0*(p[1] - p[2]) + p[3] - p[0])));
}

void interpolateCubicOther(int i, int j, Mat& src, Mat& dst, double scale) 
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
		double p[4][4] = { {src.at<Vec3b>(y_0, x_0)[k],src.at<Vec3b>(y_0, x)[k],src.at<Vec3b>(y_0, x_1)[k],src.at<Vec3b>(y_0, x_2)[k]},
		{src.at<Vec3b>(y, x_0)[k],src.at<Vec3b>(y, x)[k],src.at<Vec3b>(y, x_1)[k],src.at<Vec3b>(y, x_2)[k]},
		{src.at<Vec3b>(y_1, x_0)[k],src.at<Vec3b>(y_1, x)[k],src.at<Vec3b>(y_1, x_1)[k],src.at<Vec3b>(y_1, x_2)[k]},
		{src.at<Vec3b>(y_2, x_0)[k],src.at<Vec3b>(y_2, x)[k],src.at<Vec3b>(y_2, x_1)[k],src.at<Vec3b>(y_2, x_2)[k]} };

		double arr[4];
		arr[0] = cubicInterpolate(p[0], deltaX);
		arr[1] = cubicInterpolate(p[1], deltaX);
		arr[2] = cubicInterpolate(p[2], deltaX);
		arr[3] = cubicInterpolate(p[3], deltaX);

		double res = cubicInterpolate(arr, deltaY);
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
			interpolateLinear(i, j, src, dst, scale);
		}
	}
}

void resizeCubic(Mat& src, Mat& dst, double scale)
{
	for (int i = 0; i < dst.rows; ++i)
	{
		for (int j = 0; j < dst.cols; ++j)
		{
			interpolateCubicOther(i, j, src, dst, scale);
		}
	}
}

void resizeMixed(Mat& src, Mat& dst, double scale)
{
	Mat gradient(src.size(), CV_8U);
	Laplacian(src, gradient, CV_16S, 5);

	double min, max;
	minMaxLoc(gradient, &min, &max);
	//cout << min << ' ' << max << endl;

	gradient = 255 * (gradient - min) / (max - min);
	/*minMaxLoc(gradient, &min, &max);
	cout << min << ' ' << max << endl;

	namedWindow("image", WINDOW_NORMAL);
	imshow("image", gradient);
	waitKey(0);*/

	for (int i = 0; i < dst.rows; ++i)
	{
		for (int j = 0; j < dst.cols; ++j)
		{
			int i1 = i / scale;
			int j1 = j / scale;
			if (gradient.at<short>(i1, j1) < 70)
				interpolateNearest(i, j, src, dst, scale);
			else
				if(gradient.at<short>(i1, j1) < 150)
					interpolateLinear(i, j, src, dst, scale);
				else
					interpolateCubicOther(i, j, src, dst, scale);;
		}
	}
}

Mat Resize(Mat& src, double scale, InterpolationType type = NEAREST)
{
	Mat dst(src.size().width * scale, src.size().height * scale, src.type(), Scalar(0, 0, 0));
	auto start = chrono::steady_clock::now();

	/*for (int i = 0; i < dst.rows; ++i)
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
	}*/

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
		break;
	}

	auto end = chrono::steady_clock::now();
	auto diff = end - start;
	cout << chrono::duration <double, milli>(diff).count() << " ms" << endl;
	return dst;
}

Mat getControlPoints(Mat& src, double scale)
{
	Mat dst(src.size().width * scale, src.size().height * scale, src.type(), Scalar(255, 0, 0));

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

int main() 
{
	Mat img = imread("earth.jpg");
	Mat result = Resize(img, 2, MIXED);

	/*double t;
	cin >> t;
	cout << B_0_4(t) << ' ' << B_0_4(1 + t) << ' ' << B_0_4(2 + t) << ' ' << B_0_4(3 + t) << endl;*/

	namedWindow("image", WINDOW_NORMAL);
	imshow("image", result);
	waitKey(0);

	imwrite("earth_mixed_new.jpg", result);
	return 0;
}