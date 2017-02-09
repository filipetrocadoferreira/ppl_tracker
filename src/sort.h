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

struct entry_conditions
{
    cv::Point2f point;
    int cam_id ;
    float dist;
};

class sort_tracker
{
public:
    sort_tracker( std::vector<entry_conditions>  e, std::vector<entry_conditions>  l) ;

    //method to assign every detection to a tracklet (or false negative)
    void data_association(std::vector<detection>detections,  std::vector<Tracklet>& tracklets);

    //update procedure
    std::vector<Result> update(std::vector<std::vector<detection>> detections);


    //draw function
    cv::Mat draw_state();
    cv::Mat draw_state(std::vector<detection>detections);

    //parameters
    int max_age;
    int min_detections;
    float min_cost;
    float distance_to_entry;
    float distance_to_leave;
    float distance_to_begin;

    int frame_count;

    //our vector of individual tracklets
    std::vector<Tracklet> tracklets;

    std::vector<Result> active_tracklets;




private:

    std::vector<detection> missed_detections;



    //calculate intersection over union of 2 bboxes
    float cost(detection d, Tracklet t);


   std::vector<entry_conditions> entry_points;
   std::vector<entry_conditions> leave_points;

    bool entry(detection d);
    bool leave(Tracklet t);

    bool check_if_duplicated(cv::Point2f p);

    float score_appearance(cv::Mat hist1,cv::Mat hist2);



    cv::Mat room;





    //assignments matrix (to not create every cycle)
    std::vector<int> assignments;
    std::vector< std::vector<double> > costMat;

};

#endif // SORT_H
