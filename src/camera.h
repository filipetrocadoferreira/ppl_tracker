#ifndef CAMERA_H
#define CAMERA_H

/***************************************************************************
 *                                                                         *
 *   Copyright (C) %2017% by Filipe Ferreira for Meta.                     *
 *                                                                         *
 *   ftrocadoferreira@gmail.com                                            *
 *                                                                         *
 * Class : Camera - Receives videofile and calibration matrix
 *                  (relation between image and floor plane)
 *                  And Outputs detections of person in the world coordnts
 *
 *
 * Most of the image processing is made in this class. Every frame
 * the detector:
 *   (https://github.com/apennisi/fastestpedestriandetectorinthewest)
 * detects person in image.False positives are deleted based on position
 * Appearance information is also collected for every valid detection
 *                                                (rgb histogram)
 *
 * Detections from all the cameras will be then fused in tracking module
 ***************************************************************************/

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

    //Receive all the need information here:
    bool init(std::string videofile,std::string maskfile,std::string calibfile,fpdw::detector::FPDWDetector * det_,cv::Point2f position);

    //Process next frame. Returns false when videofile is empty.
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

    //pointer to detector (shared among cameras)
    fpdw::detector::FPDWDetector *detector;

    cv::VideoCapture capture;


    static int counter;
    int id;
    int frame_id;

    cv::Point2f world_position;

    float ratio = 0;


    //geometric constraints: mask of the floor and perspective transform image->world
    cv::Mat mask;
    cv::Mat calibMat;

    //our detections
    std::vector<detection> m_detections;


    //debug:
    bool show_results = true;
    bool verbose = false;
    bool get_ratio = true;

    std::vector<float> ratios;

    //methods:

    //runs the detector on a resized image - Returns the detections in the original size
    std::vector<detection> detect(const cv::Mat &,float factor);

    //Filter detections based on their location in the image. If their feet are not in the floor, detections turns invalid.
    void filter_by_mask(cv::Mat mask,std::vector<detection> &detections);

    //Transform location of detection to world coordinates. Also calculates location of the person relative to camera (useful in tracking)
    void get_world_coordinates(const cv::Mat &calib, std::vector<detection> &detections);

    //get confidence of detection based on size relative to distance (not being used)
    void get_confidences( std::vector<detection> &detections);

    //extract and stores appearance model for every valid detection
    void extract_appearance(std::vector<detection> &detections,cv::Mat &img);




    float get_distance_to_Camera(cv::Point2f p);

    //stores for every camera the ratio between height and distance. Returns median of the value)
    float get_statistics(detection d, std::vector<float> & ratios);


    //utils:
    cv::Point getFeet(cv::Rect);

    bool insideMask(const cv::Mat &mask,cv::Point);

    cv::Mat getHist(cv::Mat img);

    cv::Rect reduce_bbox(cv::Rect r, float h_r, float v_r);


    //Read external files:
    void getMaskfile(std::string maskfile);
    void getCalibfile(std::string calibfile);



};

#endif // CAMERA_H
