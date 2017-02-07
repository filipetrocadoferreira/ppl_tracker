#ifndef SORT_H
#define SORT_H

#include<iostream>
#include<unistd.h>


#include "tracklet.h"
#include "Hungarian.h"

#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;

struct assignment
{
    int tracklet_id;
    int detection_id;
};

class sort_tracker
{
public:
    sort_tracker();

    //method to assign every detection to a tracklet (or false negative)
    void data_association(std::vector<cv::Rect_<float> > detections,  std::vector<Tracklet>& tracklets,float iou_thresh);

    //update procedure
    std::vector<cv::Rect_<float> > update(std::vector<cv::Rect_<float> > detections);


    //parameters
    int max_age;
    int min_detections;

    int frame_count;

    //our vector of individual tracklets
    std::vector<Tracklet> tracklets;




private:



    //calculate intersection over union of 2 bboxes
    float iou(cv::Rect_<float> bb1, cv::Rect_<float> bb2);



    //assignments matrix (to not create every cycle)
    std::vector<int> assignments;
    std::vector< std::vector<double> > costMat;

};

#endif // SORT_H
