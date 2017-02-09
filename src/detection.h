#ifndef DETECTION_H
#define DETECTION_H

#include <opencv2/opencv.hpp>

class detection
{
public:
    detection();
    detection(cv::Rect bbox_,cv::Point2f point_,cv::Mat& img_,int cam_,int frame_);

    cv::Rect    bbox;
    cv::Point2f world_point;
    cv::Point2f relative_point;
    int         frame_id;
    int         cam_id;
    cv::Mat     hist;

    float confidence;

    bool        valid;


};

#endif // DETECTION_H
