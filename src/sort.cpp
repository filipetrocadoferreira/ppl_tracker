#include "sort.h"

sort_tracker::sort_tracker()
{
    //tracker parameters:
    max_age = 2;
    min_detections = 3;
    distance_to_entry = 0.45;
    distance_to_leave = 0.6;
    distance_to_begin = 1.8;

    min_cost = 0.75;

    frame_count = 0;



    //insert room points:
    wall_points.push_back(cv::Point2f(0.0,0.0));
    wall_points.push_back(cv::Point2f(0,4.25));
    wall_points.push_back(cv::Point2f(10.0,4.25));
    wall_points.push_back(cv::Point2f(10.0,0));


}

sort_tracker::~sort_tracker()
{
    Tracklet::reset_counter();

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


    //set good detections to tracklets and the bad
    for(int i = 0 ; i < assignments.size() ; i++)
    {
        if(assignments[i]>-1)
        {
            float cost = (float)costMat[i][assignments[i]];
            if(cost < min_cost)
            {
                tracklets[i].set_detections.push_back(detections[assignments[i]]);

            }
            else
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

    //then, associate data from different cameras
    missed_detections.clear();
    for(auto & cam_det  : detections)
    {
        //associate detections to tracklets:
        data_association(cam_det,tracklets);
    }




    //update trackers with correspondent detections
    for(int i = 0 ; i < tracklets.size() ; i++)
    {


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
    for (auto it = tracklets.begin(); it!=tracklets.end(); it++)
    {
        if(it->detections_streak>min_detections && !it->active)
        {
            it->set_active();

            if(first_detection)
                first_detection = false;

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
            if(  leave(*it) && it->occluded_time > max_age)
            {
                it = tracklets.erase(it);


            }
            else if(!it->active && check_if_duplicated(it->get_state()))
            {
                it = tracklets.erase(it);
            }
            else
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
    float distance = (float)norm(d.world_point-t.get_state()) - t.uncertainty; //Subtract uncertainty to distance.

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

//check if detections is near entry location
bool sort_tracker::entry(detection d)
{
    bool result = false;

    float dist =closest_wall(d.world_point);

    if(first_detection)
        dist/=2.0;




    result = dist<distance_to_entry && cv::norm(d.relative_point)<7.0 && d.confidence>35;


    return result;
}

//check if tracklet is about to leave
bool sort_tracker::leave(Tracklet t)
{

    bool result = false;

    float dist =closest_wall(t.get_state());

    if(t.active)
        dist-=t.uncertainty;

    result = ((dist)<(distance_to_leave) );



    return result;
}


//distance to closest wall
float sort_tracker::closest_wall(cv::Point2f p)
{
    //check distances to walls
    std::vector<float> dists;
    for (int i = 1; i< wall_points.size(); i++)
    {
        dists.push_back(distance_to_wall(wall_points[i],wall_points[i-1],p));
    }
    dists.push_back(distance_to_wall(wall_points[0],wall_points[wall_points.size()-1],p));

    //get minimal distance
    return *std::min_element(dists.begin(), dists.end());
}


//distance between point 'p' and a wall (w1,w2)
float sort_tracker::distance_to_wall(cv::Point2f w1, cv::Point2f w2, cv::Point2f p)
{
    float d1 = (float)fabs((w2-w1).cross(w1-p));

    float d2 = (float)cv::norm(w2-w1);

    float d  = d1/d2;

    return d;
}

bool sort_tracker::check_if_duplicated(cv::Point2f p)
{
    bool control = false;
    for(auto &t : tracklets)
    {
        float dist = (float)cv::norm(p-t.get_state());
        control = dist<distance_to_begin && dist>0.1;


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

        //draw trajectory
        if(t.active && t.trajectory.size()>2)
            for(int i = 1 ; i < t.trajectory.size() -1  ; i++)
                cv::line(room,t.trajectory[i]*(1/resolution),t.trajectory[i+1]*(1/resolution),color,1);


    }

    for(auto & m_d : missed_detections)
    {
        cv::circle(room,m_d.world_point*(1/resolution),2,cv::Scalar(155,255,155),-1);
    }

    for (int i = 1; i< wall_points.size(); i++)
    {
        cv::line(room,wall_points[i]*(1/resolution),wall_points[i-1]*(1/resolution),cv::Scalar(255,255,255),3);
    }
    cv::line(room,wall_points[0]*(1/resolution),wall_points[wall_points.size()-1]*(1/resolution),cv::Scalar(255,255,255),3);





    return room;
}

