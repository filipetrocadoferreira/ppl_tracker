#include "detection.h"

detection::detection()
{

}

detection::detection(cv::Rect bbox_, cv::Point2f point_, cv::Mat &img_, int cam_, int frame_)
{
    bbox        = bbox;
    world_point = point_;
    hist         = img_.clone();
    cam_id      = cam_;
    frame_id    = frame_;
    confidence  = 1.0;

    valid       = true;


}
