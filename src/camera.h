#ifndef CAMERA_H
#define CAMERA_H

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "detector/fpdw_detector.h"
#include "detection.h"

class Camera
{
public:
    Camera();
    ~Camera();

    bool init(std::string videofile,std::string maskfile,std::string calibfile,fpdw::detector::FPDWDetector * det_,cv::Point2f position);

    bool process();

    std::vector<detection> get_detections()
    {
        std::vector<detection> result;
        for(auto &d:m_detections)
            if(d.valid)
                result.push_back(d);

        return result;
    }

private:

    fpdw::detector::FPDWDetector *detector;

    cv::VideoCapture capture;

    static int counter;
    int id;
    int frame_id;

    cv::Point2f world_position;

    float ratio = 0;


    cv::Mat mask;
    cv::Mat calibMat;

    std::vector<detection> m_detections;

    bool show_results = true;
    bool verbose = false;
    bool get_ratio = true;

    std::vector<float> ratios;

    //methods:
    std::vector<detection> detect(const cv::Mat &,float factor);

    void filter_by_mask(cv::Mat mask,std::vector<detection> &detections);

    void get_world_coordinates(const cv::Mat &calib, std::vector<detection> &detections);

    void get_confidences( std::vector<detection> &detections);

    void extract_appearance(std::vector<detection> &detections,cv::Mat &img);



    float get_distance_to_Camera(cv::Point2f p);
    float get_statistics(detection d, std::vector<float> & ratios);



    //utils:

    cv::Point getFeet(cv::Rect);
    bool insideMask(const cv::Mat &mask,cv::Point);
    cv::Mat getHist(cv::Mat img);
    cv::Rect reduce_bbox(cv::Rect r, float h_r, float v_r);



    void getMaskfile(std::string maskfile);
    void getCalibfile(std::string calibfile);



};

#endif // CAMERA_H
