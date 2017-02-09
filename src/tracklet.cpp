#include "tracklet.h"

int Tracklet::counter = 0;
int Tracklet::active_counter = 0;

Tracklet::Tracklet()
{


}

Tracklet::Tracklet(detection detection_)
{
    occluded_time     = 0;
    detections        = 0;
    detections_streak = 0;
    age               = 0;
    id                = counter;
    active            = false;
    counter++;

    //Kalman init:
    kalman = cv::KalmanFilter(4,2,0);



    cv::setIdentity(kalman.measurementMatrix);
    cv::setIdentity(kalman.processNoiseCov, cv::Scalar::all(1e-1));
    cv::setIdentity(kalman.measurementNoiseCov, cv::Scalar::all(1e0));
    cv::setIdentity(kalman.errorCovPost, cv::Scalar::all(0.5));

    kalman.transitionMatrix = *(cv::Mat_<float>(4, 4)<< 1,0,0,0,   0,1,0,0,  0,0,1,0,  0,0,0,1); //simplest model. Velocity is not being used because of noisy environment



    //initial state of the kalman filter will be our initial detection
    kalman.statePost.at<float>(0) = detection_.world_point.x;
    kalman.statePost.at<float>(1) = detection_.world_point.y; //center point

    kalman.statePost.at<float>(2) = 0;                 //vx=0
    kalman.statePost.at<float>(3) = 0;                //vy=0


    detectionMat = cv::Mat::zeros(2,1,CV_32FC1);

    active_camera = detection_.cam_id;

    hist = detection_.hist;

}

Tracklet::~Tracklet()
{

}

void Tracklet::update()
{

    if(set_detections.size()>0)
        {
            std::cout << " our set of detections: " << std::endl;
            for(auto & d : set_detections)
                {
                    std::cout << " Cam_id" << d.cam_id << " point " << d.world_point << std::endl;
                }




            //aggregate detections to virtual detections:
            fuse_detections();

            //initial state of the kalman filter will be our initial detection
            detectionMat.at<float>(0) = virtual_detection.world_point.x;
            detectionMat.at<float>(1) = virtual_detection.world_point.y; //center point



            //update kalman filter state with observation
            kalman.correct(detectionMat);

            //update control variables
            occluded_time = 0;
            detections++;
            detections_streak++;


        }

}

void Tracklet::predict()
{
    cv::Mat predictionMat = kalman.predict();



    //update control variables

    if(occluded_time>0)
        detections_streak=0;

    occluded_time++;
    age ++;



}

cv::Point2f Tracklet::get_state()
{
    cv::Mat stateMat = kalman.statePost;

    cv::Mat confidenceMat = kalman.errorCovPost;




    world_position = cv::Point2f(stateMat.at<float>(0),stateMat.at<float>(1));
    uncertainty =confidenceMat.at<float>(0,0);




    return world_position;
}

Result Tracklet::get_result( )
{
    Result r;
    r.world_position = get_state();

    r.id             = active_id;
    r.active_camera  = active_camera;

    r.history        = history;
    r.uncertainty    = uncertainty;

    return r;
}

void Tracklet::set_active()
{
    active_id = active_counter++;
    active = true;
}

void Tracklet::reset_counter()
{
    counter = 0;
    active_counter = 0;
}

void Tracklet::fuse_detections()
{


    if(set_detections.size()>3) //exclude the outsider
        {

            exclude_outliers();
        }

    cv::Point2f cumulative(0,0);
    float sum=0;


    //wheighted average of detections taking in account distance to camera (confidence)
    for( auto &d : set_detections)
        {
            float dist = (float)cv::norm(d.relative_point);
            cumulative = cumulative + d.world_point * (1/dist)*(1/dist);
            sum+=(1/dist)*(1/dist);
        }

    cv::Point2f result = cumulative * (1 / sum);



    virtual_detection.world_point = result;

}

void Tracklet::exclude_outliers()
{
    float diff;
    std::vector<float> diffs;
    std::vector<float> mean_diffs;


    for(int i = 0 ; i<set_detections.size() ; i++)
        {
            diffs.clear();

            for(int j = 0; j < set_detections.size(); j++)
                {
                    if(i!=j)
                        {
                            diffs.push_back((float)cv::norm(set_detections[i].world_point-set_detections[j].world_point));
                        }

                }
            //calculate mean of distances to other points
            double sum = std::accumulate(diffs.begin(), diffs.end(), 0.0);
            mean_diffs.push_back((float)(sum / diffs.size()));

        }

    float mean = (float) std::accumulate(mean_diffs.begin(), mean_diffs.end(), 0.0)/(float)mean_diffs.size();
    mean*=1.3; //our limit to delete

    int k = -1;
    for(int i = 0 ; i < mean_diffs.size(); i++)
        {
            if(mean_diffs[i]>mean)
                {
                    k = i;
                    mean = mean_diffs[i];
                }
        }

    if(k>-1)
        set_detections.erase(set_detections.begin()+k);

}
