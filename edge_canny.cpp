#define _USE_MATH_DEFINES
#include <opencv2/opencv.hpp>
#include <math.h>
#include<time.h> 

using namespace cv;
using namespace std;


void getGaussianMask(Mat& mask, Size ksize, double sigma, int threshold)
{
    if (ksize.width % 2 == 0 || ksize.height % 2 == 0)
    {
        cout << "please input odd ksize!" << endl;
        exit(-1);
    }
    if (!threshold)
    {
        mask.create(ksize, CV_64F);
        int h = ksize.height;
        int w = ksize.width;
        int center_h = (ksize.height - 1) / 2;
        int center_w = (ksize.width - 1) / 2;
        double sum = 0;
        double x, y;
        for (int i = 0; i < h; i++)
        {
            x = pow(i - center_h, 2);
            for (int j = 0; j < w; j++)
            {
                y = pow(j - center_w, 2);
                mask.at<double>(i, j) = exp(-(x + y) / (2 * sigma * sigma));
                sum += mask.at<double>(i, j);
            }
        }
        mask = mask / sum;
    }
    else
    {
        if (threshold < 2 || threshold > 255)
        {
            cout << "wrong imput" << endl;
            exit(-1);
        }
        mask.create(ksize.height, 255 + 1, CV_64F);
        float fv = threshold * 2.5;
        int i;
        mask.at<double>(0, 0) = 1;
        for (i = 1; i < 256; i++)
        {
            mask.at<double>(0, i) = 1 - i / fv;
            if (mask.at<double>(0, i) < 0) break;
        }
        for (; i < 256; i++)
            mask.at<double>(0, i) = 0;
    }
}

void myGaussianBlur(const Mat src, Mat& dst, Mat mask, int threshold)
{
    if (!threshold)
    {
        int hh = (mask.rows - 1) / 2;
        int hw = (mask.cols - 1) / 2;
        dst = Mat::zeros(src.size(), src.type());

        //边界填充
        Mat newsrc;
        copyMakeBorder(src, newsrc, hh, hh, hw, hw, BORDER_DEFAULT);
        //高斯滤波
        for (int i = hh; i < src.rows + hh; i++)
        {
            for (int j = hw; j < src.cols + hw; j++)
            {
                double sum[3] = { 0 };
                for (int r = -hh; r <= hh; r++)
                {
                    for (int c = -hw; c <= hw; c++)
                    {
                        if (src.channels() == 1)
                        {
                            sum[0] += newsrc.at<uchar>(i + r, j + c) * mask.at<double>(r + hh, c + hw);
                        }
                        else if (src.channels() == 3)
                        {
                            sum[0] += newsrc.at<Vec3b>(i + r, j + c)[0] * mask.at<double>(r + hh, c + hw);
                            sum[1] += newsrc.at<Vec3b>(i + r, j + c)[1] * mask.at<double>(r + hh, c + hw);
                            sum[2] += newsrc.at<Vec3b>(i + r, j + c)[2] * mask.at<double>(r + hh, c + hw);
                        }
                    }
                }
                for (int k = 0; k < src.channels(); k++)
                {
                    if (sum[k] < 0)sum[k] = 0;
                    else if (sum[k] > 255)sum[k] = 255;
                }
                if (src.channels() == 1)
                {
                    dst.at<uchar>(i - hh, j - hw) = static_cast<uchar>(sum[0]);
                }
                else if (src.channels() == 3)
                {
                    Vec3b rgb = { static_cast<uchar>(sum[0]) ,static_cast<uchar>(sum[1]) ,static_cast<uchar>(sum[2]) };
                    dst.at<Vec3b>(i - hh, j - hw) = rgb;
                }
            }
        }
    }
    else
    {
        int hh = (mask.rows - 1) / 2;
        int hw = (mask.rows - 1) / 2;
        dst = Mat::zeros(src.size(), src.type());

        //边界填充
        Mat newsrc;
        copyMakeBorder(src, newsrc, hh, hh, hw, hw, BORDER_DEFAULT);
        for (int i = hh; i < src.rows + hh; i++)
        {
            for (int j = hw; j < src.cols + hw; j++)
            {
                double sum[2] = { 0 };
                for (int r = -hh; r <= hh; r++)
                {
                    for (int c = -hw; c <= hw; c++)
                    {
                        int diff = abs(newsrc.at<uchar>(i + r, j + c) - newsrc.at<uchar>(i, j));
                        sum[0] += newsrc.at<uchar>(i + r, j + c) * mask.at<double>(0, diff);
                        sum[1] += mask.at<double>(0, diff);
                    }
                }
                sum[0] /= sum[1];
                if (sum[0] < 0)sum[0] = 0;
                else if (sum[0] > 255)sum[0] = 255;
                dst.at<uchar>(i - hh, j - hw) = static_cast<uchar>(sum[0]);
            }
        }
    }
}

void getGrandient(Mat img, Mat& gradXY, Mat& theta, int flag)
{
    gradXY = Mat::zeros(img.size(), CV_8U);
    theta = Mat::zeros(img.size(), CV_8U);
    if (flag == 5)
    {
        for (int i = 2; i < img.rows - 2; i++)
        {
            for (int j = 2; j < img.cols - 2; j++)
            {
                double gradY = 1 / 8 * double(img.ptr<uchar>(i - 2)[j - 2] + 4 * img.ptr<uchar>(i - 2)[j - 1] + 6 * img.ptr<uchar>(i - 2)[j] + 4 * img.ptr<uchar>(i - 2)[j + 1] + img.ptr<uchar>(i - 2)[j + 2]
                    + 2 * img.ptr<uchar>(i - 1)[j - 2] + 8 * img.ptr<uchar>(i - 1)[j - 1] + 12 * img.ptr<uchar>(i - 1)[j] + 8 * img.ptr<uchar>(i - 1)[j + 1] + 2 * img.ptr<uchar>(i - 1)[j + 2]
                    - 2 * img.ptr<uchar>(i + 1)[j - 2] - 8 * img.ptr<uchar>(i + 1)[j - 1] - 12 * img.ptr<uchar>(i + 1)[j] - 8 * img.ptr<uchar>(i + 1)[j + 1] - 2 * img.ptr<uchar>(i + 1)[j + 2]
                    - img.ptr<uchar>(i + 2)[j - 2] - 4 * img.ptr<uchar>(i + 2)[j - 1] - 6 * img.ptr<uchar>(i + 2)[j] - 4 * img.ptr<uchar>(i + 2)[j + 1] - img.ptr<uchar>(i + 2)[j + 2]);
                double gradX = 1 / 8 * double(img.ptr<uchar>(i - 2)[j - 2] + 4 * img.ptr<uchar>(i - 1)[j - 2] + 6 * img.ptr<uchar>(i)[j - 2] + 4 * img.ptr<uchar>(i + 1)[j - 2] + img.ptr<uchar>(i + 2)[j - 2]
                    + 2 * img.ptr<uchar>(i - 2)[j - 1] + 8 * img.ptr<uchar>(i - 1)[j - 1] + 12 * img.ptr<uchar>(i)[j - 1] + 8 * img.ptr<uchar>(i + 1)[j - 1] + 2 * img.ptr<uchar>(i + 2)[j - 1]
                    - 2 * img.ptr<uchar>(i - 2)[j + 1] - 8 * img.ptr<uchar>(i - 1)[j + 1] - 12 * img.ptr<uchar>(i)[j + 1] - 8 * img.ptr<uchar>(i + 1)[j + 1] - 2 * img.ptr<uchar>(i + 2)[j + 1]
                    - img.ptr<uchar>(i - 2)[j + 2] - 4 * img.ptr<uchar>(i - 1)[j + 2] - 6 * img.ptr<uchar>(i)[j + 2] - 4 * img.ptr<uchar>(i + 1)[j + 2] - img.ptr<uchar>(i + 2)[j + 2]);
                gradXY.ptr<uchar>(i)[j] = sqrt(gradX * gradX + gradY * gradY);
                theta.ptr<uchar>(i)[j] = atan(gradY / gradX);
            }
        }
    }
    if (flag == 1)
    {
        for (int i = 1; i < img.rows - 1; i++)
        {
            for (int j = 1; j < img.cols - 1; j++)
            {
                double gradY = double(img.ptr<uchar>(i - 1)[j - 1] + 10 / 3 * img.ptr<uchar>(i - 1)[j] + img.ptr<uchar>(i - 1)[j + 1] - img.ptr<uchar>(i + 1)[j - 1] - 10 / 3 * img.ptr<uchar>(i + 1)[j] - img.ptr<uchar>(i + 1)[j + 1]);
                double gradX = double(img.ptr<uchar>(i - 1)[j + 1] + 10 / 3 * img.ptr<uchar>(i)[j + 1] + img.ptr<uchar>(i + 1)[j + 1] - img.ptr<uchar>(i - 1)[j - 1] - 10 / 3 * img.ptr<uchar>(i)[j - 1] - img.ptr<uchar>(i + 1)[j - 1]);
                gradXY.ptr<uchar>(i)[j] = sqrt(gradX * gradX + gradY * gradY);
                theta.ptr<uchar>(i)[j] = atan(gradY / gradX);
            }
        }
    }
    else
    {
        for (int i = 1; i < img.rows - 1; i++)
        {
            for (int j = 1; j < img.cols - 1; j++)
            {
                double gradY = double(img.ptr<uchar>(i - 1)[j - 1] + 2 * img.ptr<uchar>(i - 1)[j] + img.ptr<uchar>(i - 1)[j + 1] - img.ptr<uchar>(i + 1)[j - 1] - 2 * img.ptr<uchar>(i + 1)[j] - img.ptr<uchar>(i + 1)[j + 1]);
                double gradX = double(img.ptr<uchar>(i - 1)[j + 1] + 2 * img.ptr<uchar>(i)[j + 1] + img.ptr<uchar>(i + 1)[j + 1] - img.ptr<uchar>(i - 1)[j - 1] - 2 * img.ptr<uchar>(i)[j - 1] - img.ptr<uchar>(i + 1)[j - 1]);
                gradXY.ptr<uchar>(i)[j] = sqrt(gradX * gradX + gradY * gradY);
                theta.ptr<uchar>(i)[j] = atan(gradY / gradX);
            }
        }
    }
}

void nonLocalMaxValue(Mat gradXY, Mat theta, Mat& dst)
{
    dst = gradXY.clone();
    for (int i = 1; i < gradXY.rows - 1; i++)
    {
        for (int j = 1; j < gradXY.cols - 1; j++)
        {
            double t = double(theta.ptr<uchar>(i)[j]);
            double g = double(dst.ptr<uchar>(i)[j]);
            if (g == 0.0)
                continue;
            double g0, g1;
            if ((t >= -(3 * M_PI / 8)) && (t < -(M_PI / 8)))
            {
                g0 = double(dst.ptr<uchar>(i - 1)[j - 1]);
                g1 = double(dst.ptr<uchar>(i + 1)[j + 1]);
            }
            else if ((t >= -(M_PI / 8)) && (t < M_PI / 8))
            {
                g0 = double(dst.ptr<uchar>(i)[j - 1]);
                g1 = double(dst.ptr<uchar>(i)[j + 1]);
            }
            else if ((t >= M_PI / 8) && (t < 3 * M_PI / 8))
            {
                g0 = double(dst.ptr<uchar>(i - 1)[j + 1]);
                g1 = double(dst.ptr<uchar>(i + 1)[j - 1]);
            }
            else
            {
                g0 = double(dst.ptr<uchar>(i - 1)[j]);
                g1 = double(dst.ptr<uchar>(i + 1)[j]);
            }
            if (g <= g0 || g <= g1) {
                dst.ptr<uchar>(i)[j] = 0.0;
            }
        }
    }
}

bool bondCheck(Mat mat, int i, int j)
{
    if (i < 0 || j < 0 || i >= mat.rows || j >= mat.cols)
        return false;
    return true;
}

void linkLine(Mat img, Mat& flag, int i, int j)
{
    if (img.ptr<uchar>(i)[j] != 0 && flag.ptr<uchar>(i)[j] == 0)
    {
        uchar f = 255;
        flag.ptr<uchar>(i)[j] = f;
        for (int m = -1; m < 1; m++)
        {
            for (int n = -1; n < 1; n++)
            {
                if (bondCheck(img, i + m, j + n))
                {
                    linkLine(img, flag, i + m, j + n);
                }
            }
        }
    }
    else if (img.ptr<uchar>(i)[j] == 0)
        flag.ptr<uchar>(i)[j] = 1;
}


void doubleThresholdLink(Mat img,Mat& dst)
{
    dst=img.clone();
    int c = 1;
    srand(time(nullptr));
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            if (img.ptr<uchar>(i)[j] == 255 && dst.ptr<uchar>(i)[j] == 0)
            {
                linkLine(img, dst, i, j);
            }
            else if (img.ptr<uchar>(i)[j] == 0)
                dst.ptr<uchar>(i)[j] = 1;
        }
    }
}

void doubleThreshold(double low, double high, Mat img, Mat& dst)
{
    dst = Mat::zeros(Size(img.cols, img.rows), CV_8UC3);
    for (int i = 0; i < img.rows - 1; i++)
    {
        for (int j = 0; j < img.cols - 1; j++)
        {
            double x = double(img.ptr<uchar>(i)[j]);
            if (x > high)
                img.ptr<uchar>(i)[j] = 255;
            else if (x < low)
                img.ptr<uchar>(i)[j] = 0;
            else
                img.ptr<uchar>(i)[j] = 120;
        }
    }
    doubleThresholdLink(img, dst);
}

void mergeImg(Mat& dst, Mat src1, Mat src2) {
    int rows = src1.rows;
    int cols = src1.cols + src2.cols;
    CV_Assert(src1.type() == src2.type());
    dst.create(rows, cols, src1.type());
    src1.copyTo(dst(Rect(0, 0, src1.cols, src1.rows)));
    src2.copyTo(dst(Rect(src1.cols, 0, src2.cols, src2.rows)));
}

int main() {

    // 加载灰度图像
    /*String path;
    cin >> path;
    path += ".png";*/

    string str_dir = "C:\\Users\\11634\\Desktop\\论文3";
    vector<string> imgs;
    //得到文件夹里所有图片路径
    glob(str_dir, imgs, true);
    for (int img_id = 0; img_id < imgs.size(); ++img_id)
    {
        string str_img = imgs[img_id];
        Mat img  = imread(str_img, IMREAD_GRAYSCALE);
        // Mat img = imread(path, IMREAD_GRAYSCALE);

        if (img.empty()) {
            printf("Empty path");
            return 0;
        }

        int threshold=0, flag=3;
        // cin >> threshold >> flag;

        // 高斯滤波
        Mat mask;
        getGaussianMask(mask, Size(5, 5), 0.8, threshold);
        Mat gauss_img;
        myGaussianBlur(img, gauss_img, mask, threshold);

        Mat outImg;
        mergeImg(outImg, img, gauss_img);
        namedWindow("img");
        imshow("img", outImg);
        //waitKey();

        // 计算梯度幅值和方向
        Mat gradXY, theta;
        getGrandient(gauss_img, gradXY, theta, flag);

        // 局部非极大值抑制
        Mat local_img;
        nonLocalMaxValue(gradXY, theta, local_img);

        // 双阈值算法检测和连接边缘
        Mat dst;
        doubleThreshold(50, 160, local_img, dst);

        // 输出
        mergeImg(outImg, img, dst);
        namedWindow("img");
        imshow("img", outImg);
        string path = to_string(img_id+20)+ "edge.jpg";
        imwrite(path, dst);
        //waitKey(0);
    }
    return 0;
}