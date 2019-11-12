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

	//TO DO
	
	return 0;
}

void resizeNearest(Mat& src, Mat& dst, double scale)
{
	for (int i = 0; i < dst.rows; ++i) 
	{
		for (int j = 0; j < dst.cols; ++j)
		{
			int i1 = i / scale;
			int j1 = j / scale;
			dst.at<Vec3b>(i, j) = src.at<Vec3b>(i1, j1);
		}
	}
}

void resizeLinear(Mat& src, Mat& dst, double scale)
{
	/*Mat exp(src.rows + 2, src.cols + 2, src.type());
	expandImage(src, exp);*/
	//src.convertTo(src, CV_32FC3);

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
			double deltaY = i1 - y;
			double deltaX = j1 - x;
		
			auto f_up = src.at<Vec3b>(y, x) * B_0_2(deltaX + 1) + src.at<Vec3b>(y, x_1) * B_0_2(deltaX);
			auto f_down = src.at<Vec3b>(y_1, x) * B_0_2(deltaX + 1) + src.at<Vec3b>(y_1, x_1) * B_0_2(deltaX);

			auto res = f_up * B_0_2(deltaY + 1) + f_down * B_0_2(deltaY);
			dst.at<Vec3b>(i, j) = res;
		}
	}
}

void resizeCubic(Mat& src, Mat& dst, double scale)
{

}

void resizeMixed(Mat& src, Mat& dst, double scale)
{

}


Mat Resize(Mat& src, double scale, InterpolationType type = NEAREST)
{
	Mat dst(src.size().width * scale, src.size().height * scale, src.type(), Scalar(0, 0, 0));
	//auto start = chrono::steady_clock::now();
	//resize(src, dst, dst.size(), 0, 0, INTER_LANCZOS4);

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

	// auto end = chrono::steady_clock::now();
	// auto diff = end - start;
	// cout << chrono::duration <double, milli>(diff).count() << " ms" << endl;
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

	Mat result = Resize(img, 2, BILINEAR);


	namedWindow("image", WINDOW_NORMAL);
	imshow("image", result);
	waitKey(0);

	imwrite("earth_linear_new.jpg", result);
	return 0;
}