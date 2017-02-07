#include "sort.h"

sort_tracker::sort_tracker()
{
    //tracker parameters:
    max_age = 2;
    min_detections = 5;

    frame_count = 0;
}

void sort_tracker::data_association(std::vector<cv::Rect_<float> > detections, std::vector<Tracklet>& tracklets, float iou_thresh)
{


    int n_detections = detections.size();
    int n_tracklets  = tracklets.size();

    //reset matrix:
    assignments.clear();
    costMat.clear();
    costMat.resize(n_tracklets,std::vector<double>(n_detections,0));



    //insert the values in the assigning cost matrix
    for(int i = 0; i< n_tracklets; i++)
        for(int j = 0; j< n_detections; j++)
        {

            costMat[i][j] =(double)((1.0-iou(tracklets[i].get_state(),detections[j]))); //convert to int because of the solver
        }




    //hungarian algorithm for assigning
    HungarianAlgorithm hungarian;
    if(costMat.size()>0)
        hungarian.Solve(costMat,assignments);

}

std::vector<cv::Rect_<float> > sort_tracker::update(std::vector<cv::Rect_<float> > detections)
{
    //first : predict

    for(auto &t : tracklets)
    {
        t.predict();
    }


    //associate detections to tracklets:
    data_association(detections,tracklets,0.3);

    //collect unmatched detections:
    std::vector<int> unmatched_detections;
    for(int i = 0; i< detections.size(); i++)
    {
        int idx = -1;
        for (int j = 0; j< assignments.size(); j++)
        {
            if(i == assignments[j])
            {
                idx = 0;
                continue;
            }
        }
        if(idx < 0)
            unmatched_detections.push_back(i);
    }





    //update trackers with correspondent detections
    for(int i = 0 ; i < tracklets.size() ; i++)
    {
        if(assignments[i]!=-1)
        {
            tracklets[i].update(detections[assignments[i]]);
        }
    }

    //init new tracklets from unmatched detections
    for(int &u_d : unmatched_detections)
    {
        tracklets.push_back(Tracklet(detections[u_d]));
    }

    std::vector<cv::Rect_<float> > result;


    //return active tracklets and delete the old ones
    for (auto it = tracklets.begin(); it!=tracklets.end();)
    {
        if(it->detections_streak>min_detections)
        {
            result.push_back(it->get_state());
        }


        //delete old tracklets
        it++;
        if(it!=tracklets.end())
            if( it-> occluded_time > max_age)
            {
                it = tracklets.erase(it);


            }



    }

    frame_count++;


    return result;


}

float sort_tracker::iou(cv::Rect_<float> bb1, cv::Rect_<float> bb2)
{
    float intersection = (bb1 & bb2).area();
    float reunion = bb1.area() + bb2.area() - intersection;

    if(reunion > 0)
        return (float)(intersection / reunion);
    else
        return 0.0;
}
