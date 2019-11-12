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

Mat Resize(Mat& src, double scale, InterpolationType type = BILINEAR)
{
	Mat dst(src.size().width * scale, src.size().height * scale, src.type());

	clock_t tStart = clock();

	auto start = chrono::steady_clock::now();
	resize(src, dst, dst.size(), 0, 0, INTER_LANCZOS4);
	auto end = chrono::steady_clock::now();
	auto diff = end - start;
	cout << chrono::duration <double, milli>(diff).count() << " ms" << endl;

	cout << (double)(clock() - tStart) / CLOCKS_PER_SEC << endl;

	return dst;
}

int main() 
{
	Mat img = imread("earth.jpg");

	Mat result = Resize(img, 4);

	//namedWindow("image", WINDOW_NORMAL);
	//imshow("image", result);
	//imwrite("earth_lanczos4.jpg", result);
	//waitKey(0);
	return 0;
}