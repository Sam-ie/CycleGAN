#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

const float  RATIO = 0.45;

using namespace cv;
using namespace std;


int main() {
	string path[3][2] = { {"b.jpg","a.jpg"},{"1.jpg","2.jpg"},{"church2.jpg","church1.jpg"} };
	string flag;
	cin >> flag;
	Mat box, scene;
	if (flag == "1" || flag == "2" || flag == "3")
	{
		box = imread("./image/" + path[atoi(flag.c_str()) - 1][0]);
		scene = imread("./image/" + path[atoi(flag.c_str()) - 1][1]);
	}
	else
	{
		string flag2;
		cin >> flag2;
		box = imread("./image/" + flag);
		scene = imread("./image/" + flag2);
	}

	if (!box.data || !scene.data) {
		cout << "没有找到图片" << endl;
		return 1;
	}

	//创建关键点集变量
	vector<KeyPoint> kpt_obj, kpt_sence;
	//描述子
	Mat descriptors_box, descriptors_sence;

	//计算描述符（特征向量）
	Ptr<ORB> detector = ORB::create(10000); 
	//Ptr<SIFT>detector = SIFT::create(3000);


	//检测
	detector->detectAndCompute(scene, Mat(), kpt_sence, descriptors_sence);
	detector->detectAndCompute(box, Mat(), kpt_obj, descriptors_box);
	/*detector->detect(scene, kpt_sence, Mat());
	detector->detect(box, kpt_obj, Mat());
	detector->compute(scene, kpt_sence, descriptors_sence);
	detector->compute(box, kpt_obj, descriptors_box);*/


	vector<DMatch> matches;
	//基于FLANN的描述符对象匹配
	Ptr<DescriptorMatcher> matcher = makePtr<FlannBasedMatcher>(makePtr<flann::LshIndexParams>(12, 20, 2));
	//匹配
	matcher->match(descriptors_box, descriptors_sence, matches);
	/*
	FlannBasedMatcher matcher;
	vector< DMatch > matches;
	matcher.match(descriptors_box, descriptors_sence, matches);
	*/
	cout << "Raw Matches : " << matches.size() << endl;
	// Draw Matches
	Mat outImg;
	drawMatches(box, kpt_obj, scene, kpt_sence, matches, outImg, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imshow("Raw Matches", outImg);
	waitKey(0);

	//发现匹配
	vector<DMatch> goodMatches;

	float maxdist = 0;
	//matches[i].distance描述符欧式距离（knn）

	//找到最大的距离
	for (unsigned int i = 0; i < matches.size(); ++i) {
		//printf("dist : %.2f \n", matches[i].distance);
		maxdist = max(maxdist, matches[i].distance);
	}
	for (unsigned int i = 0; i < matches.size(); ++i) {
		if (matches[i].distance < maxdist * RATIO)
			goodMatches.push_back(matches[i]);
	}

	cout << "Good Matches : " << goodMatches.size() << endl;
	Mat dst;
	drawMatches(box, kpt_obj, scene, kpt_sence, goodMatches, dst, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imshow("Good Matches", dst);
	waitKey(0);

	//---------------------------------------------------------------------------
	vector<Point2f> X;
	vector<Point2f> Y;
	X.clear();
	Y.clear();
	for (unsigned int i = 0; i < goodMatches.size(); i++) {
		int idx1 = goodMatches[i].queryIdx;
		int idx2 = goodMatches[i].trainIdx;
		X.push_back(kpt_obj[idx1].pt);
		Y.push_back(kpt_sence[idx2].pt);
	}
	vector<unsigned char> listpoints;

	Mat H = findHomography(X, Y, RANSAC, 3, listpoints);//计算透视变换

	vector<DMatch> goodgood_matches;
	for (int i = 0; i < listpoints.size(); i++)
		if ((int)listpoints[i])
			goodgood_matches.push_back(goodMatches[i]);
	cout << "Good Good Matches : " << goodgood_matches.size() << endl;
	Mat Homgimg_matches;
	drawMatches(box, kpt_obj, scene, kpt_sence,
		goodgood_matches, Homgimg_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	imshow("Good Good Matches", Homgimg_matches);
	waitKey(0);


	Mat raw1, raw2;
	vector<KeyPoint> kpt1,kpt2;
	for (int i = 0; i < goodgood_matches.size(); i++)
	{
		kpt1.push_back(kpt_obj[goodgood_matches[i].queryIdx]);
		kpt2.push_back(kpt_sence[goodgood_matches[i].trainIdx]);
	}
	drawKeypoints(box, kpt1, raw1,Scalar::all(-1));
	drawKeypoints(scene, kpt2, raw2,Scalar::all(-1));

	Mat raw = Mat(raw1.rows, raw1.cols + raw2.cols, CV_8UC3, Scalar::all(0));
	Mat ROI_1 = raw(Rect(0, 0, raw1.cols, raw1.rows));
	Mat ROI_2 = raw(Rect(raw1.cols, 0, raw2.cols, raw2.rows));
	raw1.copyTo(ROI_1);
	raw2.copyTo(ROI_2);

	imshow("Good Matching Points", raw);
	waitKey(0);

	cout << "Homography : " << endl << H << endl;

	//从待测图片中获取角点
	vector<Point2f>obj_corners(4);
	obj_corners[0] = Point(0, 0);
	obj_corners[1] = Point(box.cols, 0);
	obj_corners[2] = Point(box.cols, box.rows);
	obj_corners[3] = Point(0, box.rows);
	vector<Point2f>scene_corners(4);

	//进行透视变换
	perspectiveTransform(obj_corners, scene_corners, H);

	//绘制出角点之间的直线
	line(Homgimg_matches, scene_corners[0] + Point2f(static_cast<float>(box.cols), 0), scene_corners[1] + Point2f(static_cast<float>(box.cols), 0), Scalar(255, 0, 123), 4);
	line(Homgimg_matches, scene_corners[1] + Point2f(static_cast<float>(box.cols), 0), scene_corners[2] + Point2f(static_cast<float>(box.cols), 0), Scalar(255, 0, 123), 4);
	line(Homgimg_matches, scene_corners[2] + Point2f(static_cast<float>(box.cols), 0), scene_corners[3] + Point2f(static_cast<float>(box.cols), 0), Scalar(255, 0, 123), 4);
	line(Homgimg_matches, scene_corners[3] + Point2f(static_cast<float>(box.cols), 0), scene_corners[0] + Point2f(static_cast<float>(box.cols), 0), Scalar(255, 0, 123), 4);

	//显示最终结果
	imshow("output", Homgimg_matches);
	waitKey(0);

	//---------------------------------------------------------------------------

	return 0;
}
