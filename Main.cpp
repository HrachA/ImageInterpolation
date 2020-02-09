#include "Interpolate.h"

string name = "earthMedium";

int main()
{
	Mat img = imread(name + ".jpg");

	double scale = 2;

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
