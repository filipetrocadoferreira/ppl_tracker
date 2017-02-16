#ifndef TRACKLET_H
#define TRACKLET_H

/***************************************************************************
 *                                                                         *
 *   Copyright (C) %2017% by Filipe Ferreira for Meta.                     *
 *                                                                         *
 *   ftrocadoferreira@gmail.com                                            *
 *                                                                         *
 * Class : Tracklet - Stores information of invdividual tracklets and
 * updates internal state based on a Kalman Filter.
 *
 * Tracklet receives detections from different cameras and mix them based
 * on their confidence (distance to camera) and internal state. Outliers
 * detections are filtered clustering mutual distances.
 * Internal State is filtered using a Kalman Filter (velocity=0 model)
 * Uncertainty is calculated based on PosterioriCovariance Matrix and used
 * to assign detections.
 *
 *
 *
 ***************************************************************************/

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
    cv::Point2f get_state(); //retrieves world position
    Result get_result();

    //turn tracklet active:
    void set_active();

    //set of detections from different cameras. Outliers will be filtered. Set is mixed using a weighted mean and saved in virtual detection.
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

    std::vector <cv::Point2f> trajectory;

    bool active;

    static void reset_counter();

private:

    cv::KalmanFilter kalman;

    cv::Mat detectionMat;

    //fuse set of detections using a weight mean based on distance to camera. Detections close to camera are better than far way.
    void fuse_detections();
    //exclude (at most) one detection based on the mean of mutual distance (distance of each detection to each others)
    void exclude_outliers();

    cv::Point2f world_position;

    std::vector<cv::Point2f> history;





    static int counter;
    static int active_counter;


};

#endif // TRACKLET_H
