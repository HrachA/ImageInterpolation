#include <opencv2/opencv.hpp>
#include <iostream>
#include <ctime>

using namespace std;
using namespace cv;

string name = "earth";

enum InterpolationType
{
	NEAREST,
	BILINEAR,
	BICUBIC,
	MIXED
};


//double B_0_2(double t)
//{
//	if (0 <= t && t < 1)
//		return t;
//	if (1 <= t && t < 2)
//		return 2 - t;
//	return 0;
//}
//
//double B_0_4(double t)
//{
//	if (0 <= t && t < 1)
//		return (t * t * t) / 6;
//	if (1 <= t && t < 2)
//		return ((-3) * t * t * t + 12 * t * t - 12 * t + 4) / 6;
//	if (2 <= t && t < 3)
//		return (3 * t * t * t - 24 * t * t + 60 * t - 44) / 6;
//	if (3 <= t && t < 4)
//		return ((4 - t) * (4 - t) * (4 - t)) / 6;
//
//	return 0;
//}

void interpolateNearest(int i, int j, Mat& src, Mat& dst, double scale)
{
	int i1 = i / scale;
	int j1 = j / scale;
	dst.at<Vec3b>(i, j) = src.at<Vec3b>(i1 + 0.5, j1 + 0.5);
}

//void interpolateLinear(int i, int j, Mat& src, Mat& dst, double scale)
//{
//	double i1 = i / scale;
//	double j1 = j / scale;
//	int y = (int)i1;
//	int x = (int)j1;
//	int y_1 = y + 1 < src.rows ? y + 1 : y;
//	int x_1 = x + 1 < src.cols ? x + 1 : x;
//	double deltaY = i1 - y;
//	double deltaX = j1 - x;
//
//	auto f_up = src.at<Vec3b>(y, x) * B_0_2(deltaX + 1) + src.at<Vec3b>(y, x_1) * B_0_2(deltaX);
//	auto f_down = src.at<Vec3b>(y_1, x) * B_0_2(deltaX + 1) + src.at<Vec3b>(y_1, x_1) * B_0_2(deltaX);
//
//	auto res = f_up * B_0_2(deltaY + 1) + f_down * B_0_2(deltaY);
//	dst.at<Vec3b>(i, j) = res;
//}

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

//void interpolateCubic(int i, int j, Mat& src, Mat& dst, double scale)
//{
//	double i1 = i / scale;
//	double j1 = j / scale;
//	int y = (int)i1;
//	int x = (int)j1;
//	int y_1 = y + 1 < src.rows ? y + 1 : y;
//	int x_1 = x + 1 < src.cols ? x + 1 : x;
//	int y_2 = y_1 + 1 < src.rows ? y_1 + 1 : y_1;
//	int x_2 = x_1 + 1 < src.cols ? x_1 + 1 : x_1;
//	int y_0 = y - 1 >= 0 ? y - 1 : y;
//	int x_0 = x - 1 >= 0 ? x - 1 : x;
//	double deltaY = i1 - y;
//	double deltaX = j1 - x;
//
//
//	auto f_y_minus_1 = src.at<Vec3b>(y_0, x_0) * B_0_4(deltaX + 3) +
//						src.at<Vec3b>(y_0, x) * B_0_4(deltaX + 2) +
//						src.at<Vec3b>(y_0, x_1) * B_0_4(deltaX + 1) +
//						src.at<Vec3b>(y_0, x_2) * B_0_4(deltaX);
//
//	auto f_y = src.at<Vec3b>(y, x_0) * B_0_4(deltaX + 3) +
//						src.at<Vec3b>(y, x) * B_0_4(deltaX + 2) +
//						src.at<Vec3b>(y, x_1) * B_0_4(deltaX + 1) +
//						src.at<Vec3b>(y, x_2) * B_0_4(deltaX);
//
//	auto f_y_plus_1 = src.at<Vec3b>(y_1, x_0) * B_0_4(deltaX + 3) +
//						src.at<Vec3b>(y_1, x) * B_0_4(deltaX + 2) +
//						src.at<Vec3b>(y_1, x_1) * B_0_4(deltaX + 1) +
//						src.at<Vec3b>(y_1, x_2) * B_0_4(deltaX);
//
//	auto f_y_plus_2 = src.at<Vec3b>(y_2, x_0) * B_0_4(deltaX + 3) +
//						src.at<Vec3b>(y_2, x) * B_0_4(deltaX + 2) +
//						src.at<Vec3b>(y_2, x_1) * B_0_4(deltaX + 1) +
//						src.at<Vec3b>(y_2, x_2) * B_0_4(deltaX);
//
//
//	auto res = f_y_minus_1 * B_0_4(deltaY + 3) +
//						f_y * B_0_4(deltaY + 2) +
//						f_y_plus_1 * B_0_4(deltaY + 1) +
//						f_y_plus_2 * B_0_4(deltaY);
//
//	dst.at<Vec3b>(i, j) = res;
//}

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
	Mat mask(dst.size(), dst.type(), Scalar(0, 0, 0));

	Mat gradient;
	Mat greyMat(src.size(), CV_8U);
	cv::cvtColor(src, greyMat, cv::COLOR_BGR2GRAY);

	//    imwrite(name + "Greyscale.png", greyMat);

	Sobel(greyMat, gradient, CV_8U, 1, 1, 5);
	//    imwrite(name + "Gradient.png", gradient);
	//    Sobel(greyMat, gradient, CV_8U, 1, 0, 5);
	//    imwrite(name + "GradientX.png", gradient);
	//    Sobel(greyMat, gradient, CV_8U, 0, 1, 5);
	//    imwrite(name + "GradientY.png", gradient);
	//
	//    Laplacian(greyMat, gradient, CV_8U, 7);
	//    imwrite(name + "GradientLaplacian.png", gradient);

	//	double min, max;
	//	minMaxLoc(gradient, &min, &max);
	//	cout << min << ' ' << max << endl;

	//	gradient = 255 * (gradient - min) / (max - min);
	//

	//	minMaxLoc(gradient, &min, &max);
	//	cout << min << ' ' << max << endl;
	//
	//	namedWindow("image", WINDOW_NORMAL);
	//	imshow("image", gradient);
	//	waitKey(0);

	//    cout << gradient.at<double>(111/4, 833/4) << endl;
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
			//            if (gradient.at<uchar>(y, x) < 1 && gradient.at<uchar>(y_1, x_1) < 1 &&
			//                gradient.at<uchar>(y_1, x) < 1 && gradient.at<uchar>(y, x_1) < 1)
			//            {
			//				interpolateNearest(i, j, src, dst, scale);
			//                mask.at<Vec3b>(i, j)[0] = 255;
			//
			//            }
			//            else
			if (abs(gradient.at<uchar>(y, x)) < 100 && gradient.at<uchar>(y_1, x_1) < 100)
			{
				interpolateBilinear(i, j, src, dst, scale);
				mask.at<Vec3b>(i, j)[1] = 255;
			}
			else
			{
				interpolateBicubic(i, j, src, dst, scale);
				mask.at<Vec3b>(i, j)[2] = 255;
			}
		}
	}
	Mat res(dst.size(), dst.type());
	//    GaussianBlur(dst, res, {5, 5}, 1);
	//    dst = res;
	imwrite(name + "Mask.png", mask);
}

Mat Resize(Mat& src, double scale, InterpolationType type = NEAREST)
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

int main()
{
	Mat img = imread(name + ".jpg");

	double scale = 4;

	Mat result = Resize(img, scale, NEAREST);
	imwrite(name + std::to_string((int)scale) + "nearest.png", result);

	result = Resize(img, scale, BILINEAR);
	imwrite(name + std::to_string((int)scale) + "linear.png", result);

	result = Resize(img, scale, BICUBIC);
	imwrite(name + std::to_string((int)scale) + "cubic.png", result);

	result = Resize(img, scale, MIXED);
	imwrite(name + std::to_string((int)scale) + "mixed.png", result);

	Mat points = getControlPoints(img, scale);
	imwrite(name + std::to_string((int)scale) + "points.png", points);

	//	namedWindow("image", WINDOW_NORMAL);
	//	imshow("image", result);
	//	waitKey(0);


	return 0;
}
