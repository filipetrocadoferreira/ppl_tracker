#include "tracklet.h"

int Tracklet::counter = 0;

Tracklet::Tracklet()
{


}

Tracklet::Tracklet(cv::Rect_<float> detection)
{
     occluded_time     = 0;
     detections        = 0;
     detections_streak = 0;
     age               = 0;
     id                = counter;
    counter++;

    //Kalman init:
    kalman = cv::KalmanFilter(7,4,0);



    cv::setIdentity(kalman.measurementMatrix);
    cv::setIdentity(kalman.processNoiseCov, cv::Scalar::all(1e-2));
    cv::setIdentity(kalman.measurementNoiseCov, cv::Scalar::all(1e-1));
    cv::setIdentity(kalman.errorCovPost, cv::Scalar::all(1));
    kalman.transitionMatrix = *(cv::Mat_<float>(7, 7) <<
                                1, 0, 0, 0, 1, 0, 0,
                                0, 1, 0, 0, 0, 1, 0,
                                0, 0, 1, 0, 0, 0, 1,
                                0, 0, 0, 1, 0, 0, 0,
                                0, 0, 0, 0, 1, 0, 0,
                                0, 0, 0, 0, 0, 1, 0,
                                0, 0, 0, 0, 0, 0, 1);


    //initial state of the kalman filter will be our initial detection
    kalman.statePost.at<float>(0,0) = detection.x + detection.width/2;
    kalman.statePost.at<float>(1,0) = detection.y + detection.height/2; //center point

    kalman.statePost.at<float>(2,0) = (float)detection.area();                 //size
    kalman.statePost.at<float>(3,0) = detection.width/detection.height; //bounding box ratio


    detectionMat = cv::Mat::zeros(4,1,CV_32FC1);

}

Tracklet::~Tracklet()
{

}

void Tracklet::update(cv::Rect_<float> detection)
{

    //update measuremente matrix with detection information
    detectionMat.at<float>(0,0) = detection.x + detection.width/2;
    detectionMat.at<float>(1,0) = detection.y + detection.height/2; //center point

    detectionMat.at<float>(2,0) = (float)detection.area();          //size
    detectionMat.at<float>(3,0) = detection.width/detection.height; //bounding box ratio

    //update kalman filter state with observation
    kalman.correct(detectionMat);

    //update control variables
    occluded_time = 0;
    detections++;
    detections_streak++;

}

cv::Rect_<float> Tracklet::predict()
{
    cv::Mat predictionMat = kalman.predict();



    //update control variables

    if(occluded_time>0)
        detections_streak=0;

    occluded_time++;
    age ++;

    //return prediction (in a bounding box)
    return convert_x_to_bbox(predictionMat.at<float>(0, 0), predictionMat.at<float>(1, 0), predictionMat.at<float>(2, 0), predictionMat.at<float>(3, 0));

}

cv::Rect_<float> Tracklet::get_state()
{
    cv::Mat stateMat = kalman.statePost;

    return convert_x_to_bbox(stateMat.at<float>(0, 0), stateMat.at<float>(1, 0), stateMat.at<float>(2, 0), stateMat.at<float>(3, 0));
}

cv::Rect_<float> Tracklet::convert_x_to_bbox(float cx, float cy, float area, float ratio)
{

    float width = sqrt(area * ratio);
    float height = area/width;

    float x = cx-width/2;
    float y = cy-height/2;




    return cv::Rect_<float> (x,y,width,height);
}
