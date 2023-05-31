#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

int ain()
{
	Mat image = imread("C:/Users/11634/Desktop/����3/test_p378_.jpg");
	Mat dst, M;//MΪ�任����

	int w = image.cols;
	int h = image.rows;
	M = getRotationMatrix2D(Point2f(w / 2, h / 2), -15, 1.0);

	//C++��abs�������Ȼ֧�ֶ������͸����������汾��ʵ���ϻ��ܹ�֧�ָ�����
	double cos = abs(M.at<double>(0, 0));
	double sin = abs(M.at<double>(0, 1));

	int nw = w * cos + h * sin;
	int nh = w * sin + h * cos;

	//��ͼ�����ת����
	M.at<double>(0, 2) += (nw / 2 - w / 2);
	M.at<double>(1, 2) += (nh / 2 - h / 2);

	warpAffine(image, dst, M, Size(nw, nh), INTER_LINEAR, 0, Scalar(0, 0, 0));
	Rect m_select = Rect((nw - w) / 2, (nh - h) / 2, w, h);
	//resize(dst, dst, Size(w, h), 0, 0, INTER_AREA);
	Mat d = dst(m_select);

	// imshow("��ת��ʾ", dst);
	imwrite("./image/5_2.jpg", d);
	return 0;
}