#include <opencv2/opencv.hpp>
using namespace cv;

int main()
{
    // 1. read img
    Mat img = imread("/home/gawain/opencv_project/resources/test_image.jpg");
    if(img.empty()) 
    {
        std::cout << "img loading failed" << std::endl;
        return -1;
    }

    // 2. gray
    Mat gray;
    cvtColor(img,gray,COLOR_BGR2GRAY);
    /*
    imshow("gray img",gray);
    waitKey(0);
    //check img form
    std::cout << gray.type() << std::endl;          
    double minVal, maxVal;
    cv::Point minIdx, maxIdx;
    cv::minMaxLoc(gray, &minVal, &maxVal, &minIdx, &maxIdx);
    std::cout << "min=" << minVal << "  max=" << maxVal << std::endl;
    */
    //save gray
    imwrite("/home/gawain/opencv_project/test_image/gray img.png",gray);
    
    //3.medium filter
    Mat medianImg;
    medianBlur(gray,medianImg,5);
    //imshow("medium filter",medianImg);
    //waitKey(0);
    imwrite("/home/gawain/opencv_project/test_image/medium filter.png",medianImg);

    /* no use to the project
    //4.Canny edge check
    Mat edges;
    Canny(medianImg, edges, 100, 200);
    imshow("Canny边缘检测", edges);
    waitKey(0);
    imwrite("/home/gawain/opencv_project/test_image/edges.png",edges);
    
    
    //PS::euqalization
    Mat equalized;
    equalizeHist(medianImg, equalized);
    imwrite("/home/gawain/opencv_project/test_image/equalization.png", equalized);
    */

    //4.HSV
    Mat hsv,mask;
    cv::cvtColor(medianImg, medianImg, cv::COLOR_GRAY2BGR);
    cv::cvtColor(medianImg,hsv,COLOR_BGR2GRAY);
    inRange(hsv,Scalar(0,0,100),Scalar(180,50,105),mask);
    imwrite("/home/gawain/opencv_project/test_image/white_check.png",mask);

    //5.boundary
    Mat binary;
    threshold(mask, binary, 200, 255, THRESH_BINARY);
    imwrite("/home/gawain/opencv_project/test_image/binary.png", binary);

    // 6. 去除不连续黑斑：只保留最大黑色连通块
    Mat labels, stats, centroids;
    int nLabels = connectedComponentsWithStats(binary, labels, stats, centroids, 8, CV_32S);

    int maxLabel = 0;
    int maxArea  = 0;
    for (int i = 1; i < nLabels; ++i)          // 0 是背景（白色），跳过
    {
        int area = stats.at<int>(i, CC_STAT_AREA);
        if (area > maxArea)
        {
            maxArea  = area;
            maxLabel = i;
        }
    }

    // 把最大黑块复制回来，其余变黑→白
    Mat clean = Mat::zeros(binary.size(), CV_8UC1);
    if (maxLabel > 0)
        clean.setTo(255, labels == maxLabel);   // 最大黑块留黑（255 反色后）

    // 如果你想“黑底白字”→“白底黑字”，再反色一次
    clean.copyTo(binary);                 // 现在 binary 是白底黑字
    imwrite("/home/gawain/opencv_project/test_image/largest_black_only.png", binary);

        // 7. 在最大黑块内部继续剔除不规则小斑
    Mat innerLabels, innerStats, innerCentroids;
    int innerN = connectedComponentsWithStats(clean, innerLabels, innerStats,
                                            innerCentroids, 8, CV_32S);

    Mat finalClean = Mat::ones(clean.size(), CV_8UC1) * 255; // 全白底

    int noisePix = 1300; // 内部<200像素的小黑斑都抹掉（按图调）
    for (int i = 1; i < innerN; ++i)
    {
        int area = innerStats.at<int>(i, CC_STAT_AREA);
        if (area >= noisePix)                       // 只保留大斑
            finalClean.setTo(0, innerLabels == i);  // 涂黑
    }

    finalClean.copyTo(binary);  // 如果你想继续用 binary
    imwrite("/home/gawain/opencv_project/test_image/irregular_removed.png", finalClean);

    Mat colorEdges;
    GaussianBlur(finalClean, finalClean, Size(3,3), 0.8);
    Canny(finalClean,colorEdges,100,160);
    imwrite("/home/gawain/opencv_project/test_image/coloredge.png",colorEdges);

    
    //6.shape tools
    Mat morph;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(colorEdges, morph, MORPH_CLOSE, kernel);
    imwrite("/home/gawain/opencv_project/test_image/shape.png", morph);
    
    
    // 6. search for boundary
    std::vector<std::vector<Point>> contours;
    std::vector<Vec4i> hierarchy;
    findContours(morph, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // 7. 找所有“中间平行段 + 两头椭圆”组合
    std::vector<RotatedRect> mids;      // 中间平行段
    std::vector<RotatedRect> heads;     // 两头椭圆

    for (const auto& cnt : contours)
    {
        if (cnt.size() < 5) continue;

        // ① 先试试能不能当椭圆
        RotatedRect ell = fitEllipse(cnt);
        float ar = std::max(ell.size.width, ell.size.height) /
                std::min(ell.size.width, ell.size.height);
        if (ar > 1.5f)          // 长短轴差异明显
            heads.push_back(ell);

        // ② 再试试能不能当长方形（平行段）
        RotatedRect rect = minAreaRect(cnt);
        float ratio = std::max(rect.size.width, rect.size.height) /
                    std::min(rect.size.width, rect.size.height);
        float angle = std::abs(rect.angle);
        if (angle > 90) angle = 180 - angle;
        if (ratio > 2.0f && angle < 15.0f)   // 细长 + 近似竖直/水平
            mids.push_back(rect);
    }

    if (mids.empty() || heads.size() < 2)
    { std::cout << "中段或椭圆不足" << std::endl; return -1; }

    // 8. 先收集所有“合格三段组合”的轮廓点 + 中心
    auto angle360 = [](RotatedRect& r) {
    double a = r.angle;
    return a < 0 ? a + 180 : a;
    };
    struct Combo
    {
        int  m, e1, e2;               // 下标
        Point2f center;               // 三段整体中心
        std::vector<Point> pts16;     // 6段四角 → 16点
        double angle;                 // 中段长轴角度
    };

    std::vector<Combo> combos;
    for (int m = 0; m < mids.size(); ++m)
    for (int e1 = 0; e1 < heads.size(); ++e1)
    for (int e2 = e1 + 1; e2 < heads.size(); ++e2)
    {
        RotatedRect &mid = mids[m], &ell1 = heads[e1], &ell2 = heads[e2];

        // 角度平行（严厉 5°）
        double da1 = std::abs(angle360(mid) - angle360(ell1));
        double da2 = std::abs(angle360(mid) - angle360(ell2));
        da1 = std::min(da1, 180 - da1);
        da2 = std::min(da2, 180 - da2);
        if (da1 > 10 || da2 > 10) continue;

        // 共线（斜率差）
        Point2f cm = mid.center, c1 = ell1.center, c2 = ell2.center;
        double slopeM1 = (c1.y - cm.y) / (c1.x - cm.x + 1e-6);
        double slopeM2 = (c2.y - cm.y) / (c2.x - cm.x + 1e-6);
        if (std::abs(slopeM1 - slopeM2) > 1) continue;

        // 距离适中
        double d1 = norm(cm - c1), d2 = norm(cm - c2);
        if (d1 < 10 || d1 > 500 || d2 < 10 || d2 > 500) continue;

        // 记录轮廓点
        Combo cb;
        cb.m = m; cb.e1 = e1; cb.e2 = e2;
        cb.center = (cm + c1 + c2) / 3.0;
        cb.angle  = angle360(mid);

        auto push4 = [&](RotatedRect& r) {
            Point2f p[4]; r.points(p);
            for (int i = 0; i < 4; ++i)
                cb.pts16.emplace_back(cvRound(p[i].x), cvRound(p[i].y));
        };
        push4(mid); push4(ell1); push4(ell2);
        combos.push_back(cb);
    }

    if (combos.size() < 2)
    { std::cout << "合格组合不足 2 个" << std::endl; return -1; }

    // 9. 在 combos 里找两个最平行的组合
    int bestA = -1, bestB = -1;
    double bestParallel = 100;
    for (size_t i = 0; i < combos.size(); ++i)
    for (size_t j = i + 1; j < combos.size(); ++j)
    {
        double angleDiff = std::abs(combos[i].angle - combos[j].angle);
        angleDiff = std::min(angleDiff, 180 - angleDiff);
        if (angleDiff > 20) continue;                 // 必须平行

        // 评分：中心水平距越大越好（左右分布）
        double dx = std::abs(combos[i].center.x - combos[j].center.x);
        double dy = std::abs(combos[i].center.y - combos[j].center.y);
        double score = dx / (dy + 1);                // 越大越左右分开
        if (score < bestParallel)
        { bestParallel = score; bestA = i; bestB = j; }
    }

    if (bestA < 0)
    { std::cout << "无平行组合对" << std::endl; return -1; }

    // 10. 合并两个组合的全部角点 → 总外接矩形
    std::vector<Point> allPoints;
    allPoints.insert(allPoints.end(), combos[bestA].pts16.begin(), combos[bestA].pts16.end());
    allPoints.insert(allPoints.end(), combos[bestB].pts16.begin(), combos[bestB].pts16.end());
    Rect finalBox = boundingRect(allPoints);

    // 11. 画大框
    rectangle(img, finalBox, Scalar(0, 255, 0), 3);
    imshow("两平行组合大框", img);
    imwrite("/home/gawain/opencv_project/test_image/two_parallel_combo_box.png", img);
}


