#ifndef TRACKLET_H
#define TRACKLET_H

#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <detection.h>
using namespace std;

struct Result
{
    int id;
    cv::Point2f world_position;
    float uncertainty;

    int active_camera;

    std::vector<cv::Point2f> history;

};

class Tracklet
{
public:
    Tracklet();
    Tracklet( detection);
    ~Tracklet();

    //main functions for individual tracklets

    void update();
    void predict();

    //retrieve functions:
    cv::Point2f get_state();
    Result get_result();

    //turn tracklet active:
    void set_active();

    std::vector<detection> set_detections;
    detection           virtual_detection;


    //control variables:
    int occluded_time;
    int detections;
    int detections_streak;
    int age;
    int id;
    int active_id;
    int active_camera;
    float uncertainty;
    cv::Mat hist;

    bool active;

private:

    cv::KalmanFilter kalman;

    cv::Mat detectionMat;

    void fuse_detections();
    void exclude_outliers();

    cv::Point2f world_position;

    std::vector<cv::Point2f> history;





    static int counter;
    static int active_counter;


};

#endif // TRACKLET_H
