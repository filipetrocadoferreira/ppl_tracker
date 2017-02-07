#ifndef TRACKLET_H
#define TRACKLET_H

#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
using namespace std;

class Tracklet
{
public:
    Tracklet();
    Tracklet(cv::Rect_<float> detection);
    ~Tracklet();

    //main functions of individual tracklets
    void update(cv::Rect_<float> detection);
    cv::Rect_<float> predict();

    //retrieve functions:
    cv::Rect_<float> get_state();

    //uitlity function(move to private?)
    cv::Rect_<float> convert_x_to_bbox(float cx, float cy, float area, float ratio);



    //control variables:
    int occluded_time;
    int detections;
    int detections_streak;
    int age;
    int id;

private:

    cv::KalmanFilter kalman;

    cv::Mat detectionMat;

     static int counter;


};

#endif // TRACKLET_H
