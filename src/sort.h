#ifndef SORT_H
#define SORT_H

/***************************************************************************
 *                                                                         *
 *   Copyright (C) %2017% by Filipe Ferreira for Meta.                     *
 *                                                                         *
 *   ftrocadoferreira@gmail.com                                            *
 *                                                                         *
 * Class : sort_tracker - c++ implementation  of
 *                        https://github.com/abewley/sort
 *                       A simple online and realtime tracking algorithm
 *                       for 2D multiple object tracking in video sequences.
 *
 *
 * Tracker adapted to multi-camera environment. Receives detections and
 * returns active tracklets.
 * Every cycle : performs data association with Hungarian Algorithm and
 * updates individual tracklets with assigned detections.
 * -New tracklets are added if missed detection is near any entry point.
 * -Tracklets are set to active after a min number of detections
 * -Tracklets are deleted if they are close to leaving area
 ***************************************************************************/

#include<iostream>
#include<unistd.h>


#include "tracklet.h"
#include "Hungarian.h"

#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;


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
    ~sort_tracker();

    //method to assign every detection to a tracklet (or false negative)
    void data_association(std::vector<detection>detections,  std::vector<Tracklet>& tracklets);

    //update tracker state
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

    //set of active tracklets (output)
    std::vector<Result> active_tracklets;




private:

    //detections not assigned to any tracker. From missed detections, new tracklets are created.
    std::vector<detection> missed_detections;



    //Calculates the cost of assignment based on location and appearance(other components were tested)
    float cost(detection d, Tracklet t);


    //entry and leaving points
    std::vector<entry_conditions> entry_points;
    std::vector<entry_conditions> leave_points;

    //check if detection is close to entry point
    bool entry(detection d);
    //check if tracklet is about to leave
    bool leave(Tracklet t);

    //check if detection is a duplicated of active tracklet
    bool check_if_duplicated(cv::Point2f p);

    //calculates the similarity of 2 histograms:return 1 to 0 :1->similar : 0->not similar
    float score_appearance(cv::Mat hist1,cv::Mat hist2);



    //image of the room for debug purposes.
    cv::Mat room;

    //assignments matrix (to not create every cycle) for Hungarian Algorithm
    std::vector<int> assignments;
    std::vector< std::vector<double> > costMat;

};

#endif // SORT_H
