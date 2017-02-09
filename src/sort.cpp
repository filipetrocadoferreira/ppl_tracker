#include "sort.h"

sort_tracker::sort_tracker(std::vector<entry_conditions>  e, std::vector<entry_conditions> l)
{
    //tracker parameters:
    max_age = 1;
    min_detections = 3;
    distance_to_entry = 0.5;
    distance_to_leave = 0.95;
    distance_to_begin = 2;
    min_cost = 0.75;

    frame_count = 0;

    entry_points = e;
    leave_points = l;


}

void sort_tracker::data_association(std::vector<detection> detections, std::vector<Tracklet>& tracklets)
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

            costMat[i][j] =(double)cost(detections[j],tracklets[i]);
        }




    //hungarian algorithm for assigning
    HungarianAlgorithm hungarian;
    if(costMat.size()>0)
        hungarian.Solve(costMat,assignments);


    //set good detections to tracklets and the bad to the missed set
    for(int i = 0 ; i < assignments.size() ; i++)
    {
        if(assignments[i]>-1)
        {
            float cost = (float)costMat[i][assignments[i]];
           if(cost < min_cost)
            {
                tracklets[i].set_detections.push_back(detections[assignments[i]]);

                std::cout << " Adding detection with a cost of: " << cost << std::endl;

            }else
            {
                missed_detections.push_back(detections[assignments[i]]);
            }
        }


    }

    //add all the other missed detections:
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
           missed_detections.push_back(detections[i]);
    }




}

std::vector<Result > sort_tracker::update(std::vector<std::vector<detection> > detections)
{
    //first : predict


    for(auto &t : tracklets)
    {
        t.predict();

        t.set_detections.clear();
    }

    missed_detections.clear();
    for(auto & cam_det  : detections)
    {
        //associate detections to tracklets:
        data_association(cam_det,tracklets);
    }

    //    //collect unmatched detections:
    //    std::vector<int> unmatched_detections;
    //    for(int i = 0; i< detections.size(); i++)
    //    {
    //        int idx = -1;
    //        for (int j = 0; j< assignments.size(); j++)
    //        {
    //            if(i == assignments[j])
    //            {
    //                idx = 0;
    //                continue;
    //            }
    //        }
    //        if(idx < 0)
    //            unmatched_detections.push_back(i);
    //    }

    //    //filter detections
    //    for(int i = 0 ; i < assignments.size();i++ )
    //    {

    //        if(assignments[i]>-1)
    //        {
    //            double cost = cv::norm(tracklets[i].get_state()-detections[assignments[i]].world_point);


    //            if((float)cost>min_cost)
    //            {
    //                assignments[i]=-1;


    //            }
    //        }



    //    }



    //update trackers with correspondent detections
    for(int i = 0 ; i < tracklets.size() ; i++)
    {

        std::cout << " updating tracklet " << i << std::endl;
        tracklets[i].update();

    }

    //init new tracklets from unmatched detections
    for(auto &m_d : missed_detections)
    {
        if(entry(m_d))
        {

            tracklets.push_back(Tracklet(m_d));

            break;
        }
    }


    //reset result vector
    active_tracklets.clear();


    //return active tracklets and delete the old ones
    for (auto it = tracklets.begin(); it!=tracklets.end();it++)
    {
        if(it->detections>min_detections && !it->active)
        {
            it->set_active();

        }

        if(it->active)
        {
            active_tracklets.push_back(it->get_result());
        }
    }

    //delete old tracklets
    for (auto it = tracklets.begin(); it!=tracklets.end();)
    {
        if(it!=tracklets.end())
        {
            if(  leave(*it))
            {
                it = tracklets.erase(it);


            } else if(!it->active && check_if_duplicated(it->get_state()))
            {
                it = tracklets.erase(it);
            }else
            {
                it++;
            }
        }


    }




    frame_count++;

    return active_tracklets;



}

float sort_tracker::cost(detection d, Tracklet t)
{
    //first component: distance
    float distance = (float)norm(d.world_point-t.get_state()) - t.uncertainty; //should we deal with uncertainty? sure.... later

    //second component: proximity to camera
    float proximity = (float)norm(d.relative_point);

    //third component : confidence:
    float confidence = (float)fabs(d.confidence);

    //4th component : appearance
    float appearance = 1.0 - score_appearance(d.hist,t.hist);

    //then we mix all together (how? hmmm)
    float total = distance *( 1+appearance );



    if (total<0)
        total = 0;



    return total;
}

bool sort_tracker::entry(detection d)
{
    bool result = false;
    for (auto &c : entry_points)
    {
     float dist =cv::norm(d.world_point-c.point);

     result = (dist<c.dist && d.cam_id==c.cam_id);

     if(result)
         break;
    }



   return result;
}

bool sort_tracker::leave(Tracklet t)
{

    bool result = false;
    for (auto &c : leave_points)
    {
     float dist =cv::norm(t.get_state()-c.point);

     result = (dist<(c.dist*0.8) );

     if(result)
         break;
    }
    return result;
}

bool sort_tracker::check_if_duplicated(cv::Point2f p)
{
    bool control = false;
    for(auto &t : active_tracklets)
    {
        control = (float)cv::norm(p-t.world_position)<distance_to_begin;


    }

    return control;
}

float sort_tracker::score_appearance(cv::Mat hist1, cv::Mat hist2)
{
    if(hist1.empty() || hist2.empty())
        return 1.0;

    double score = cv::compareHist( hist1, hist2, 0);

    return (float)score;
}

cv::Mat sort_tracker::draw_state()
{
    //draw Tracker information
    int max_x = 10;
    int max_y = 5;

    float resolution = 0.01;

    int width = max_x / resolution;
    int height = max_y / resolution;

    room = cv::Mat::zeros(height,width,CV_8UC3);

    for(auto &t : tracklets)
    {
        cv::Scalar color(0,0,255);
        if(t.active)
        {
            int i = t.active_id+1;
            color = cv::Scalar(255*(i%4),255*(i%3),255*(i%2));
        }
        cv::circle(room,t.get_state()*(1/resolution),3,color,-1);
        //draw uncertainty;
        float circle_size = t.uncertainty/resolution;

        cv::circle(room,t.get_state()*(1/resolution),circle_size,color,1);

        for(auto &d : t.set_detections)
        {
            cv::circle(room,d.world_point*(1/resolution),3,cv::Scalar(255,255,255),1);

            cv::line(room,d.world_point*(1/resolution),t.get_state()*(1/resolution),cv::Scalar(0,0,150),2);

        }


    }

    for(auto & m_d : missed_detections)
    {
         cv::circle(room,m_d.world_point*(1/resolution),2,cv::Scalar(155,255,155),-1);
    }

    //draw entry point
    cv::circle(room,entry_points[0].point*(1/resolution),entry_points[0].dist/resolution,cv::Scalar(0,255,0),1);

    //draw entry point
    cv::circle(room,leave_points[0].point*(1/resolution),entry_points[0].dist/resolution,cv::Scalar(0,255,255),1);



    return room;
}

cv::Mat sort_tracker::draw_state(std::vector<detection> detections)
{



    float resolution = 0.01;

    cv::Mat r = draw_state();

    for(auto &d : detections)
    {
        cv::circle(r,d.world_point*(1/resolution),3,cv::Scalar(255,255,255),1);

    }

    return r;
}


