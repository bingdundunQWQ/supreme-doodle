#include <opencv2/opencv.hpp>
using namespace cv;

int main(){
    Mat img = imread("/home/gawain/opencv_project/resources/test_image.png");
    if(img.empty()) return -1;

    // 1. 灰度 + 保存
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    imwrite("/home/gawain/opencv_project/build/out_gray.png", gray);

    // 2. HSV + 保存
    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV);
    imwrite("/home/gawain/opencv_project/build/out_hsv.png", hsv);

    // 3. 滤波 trio + 保存
    Mat blurImg, gauImg, medImg;
    blur(img, blurImg, Size(5,5));
    GaussianBlur(img, gauImg, Size(5,5), 1.5);
    medianBlur(img, medImg, 5);
    imwrite("/home/gawain/opencv_project/build/out_blur.png", blurImg);
    imwrite("/home/gawain/opencv_project/build/out_gaussian.png", gauImg);
    imwrite("/home/gawain/opencv_project/build/out_median.png", medImg);

    // 4. 红色掩膜 + 保存
    Mat mask1, mask2, redMask;
    inRange(hsv, Scalar(0,100,100),  Scalar(10,255,255),  mask1);
    inRange(hsv, Scalar(170,100,100),Scalar(180,255,255), mask2);
    redMask = mask1 | mask2;
    imwrite("/home/gawain/opencv_project/build/out_redMask.png", redMask);

    // 5. 形态学开闭 + 保存
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5,5));
    morphologyEx(redMask, redMask, MORPH_OPEN,  kernel);
    morphologyEx(redMask, redMask, MORPH_CLOSE, kernel);
    imwrite("/home/gawain/opencv_project/build/out_morph.png", redMask);

    // 6. 用连通分量替代 findContours —— 无需 vector
    Mat labels, stats, centroids;
    int n = connectedComponentsWithStats(redMask, labels, stats, centroids, 8, CV_32S);
    Mat boxImg = img.clone();
    for(int i = 1; i < n; ++i){          // 0 是背景
        int area = stats.at<int>(i, CC_STAT_AREA);
        if(area < 300) continue;
        printf("红色块 %d 面积 = %d px²\n", i, area);
        int x = stats.at<int>(i, CC_STAT_LEFT);
        int y = stats.at<int>(i, CC_STAT_TOP);
        int w = stats.at<int>(i, CC_STAT_WIDTH);
        int h = stats.at<int>(i, CC_STAT_HEIGHT);
        rectangle(boxImg, Rect(x,y,w,h), Scalar(0,0,255), 2);
    }
    imwrite("/home/gawain/opencv_project/build/out_box.png", boxImg);

    // 7. 高亮二值化 + 漫水 + 保存
    Mat binBright, flood;
    threshold(gray, binBright, 180, 255, THRESH_BINARY);
    dilate(binBright, binBright, kernel);
    erode(binBright,  binBright, kernel);
    flood = binBright.clone();
    floodFill(flood, Point(0,0), 255);
    flood = 255 - flood;
    imwrite("/home/gawain/opencv_project/build/out_bright.png", binBright);
    imwrite("/home/gawain/opencv_project/build/out_flood.png",  flood);

    // 8. 旋转35° + 保存
    Mat rotMat = getRotationMatrix2D(Point(img.cols/2,img.rows/2), 35, 1.0);
    Mat rotated;
    warpAffine(img, rotated, rotMat, img.size());
    imwrite("/home/gawain/opencv_project/build/out_rotated.png", rotated);

    // 9. 裁剪左上1/4 + 保存
    Mat crop(img, Rect(0, 0, img.cols/2, img.rows/2));
    imwrite("/home/gawain/opencv_project/build/out_crop.png", crop);

    return 0;
}


