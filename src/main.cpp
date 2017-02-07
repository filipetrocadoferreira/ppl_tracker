#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "detector/fpdw_detector.h"
#include "sort.h"
using namespace cv;

int main(int argc, char** argv )
{
    if ( argc != 3 )
    {
        std::cout << argc << std::endl;
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }

    string filename = argv[1];
    VideoCapture capture(filename);


    if( !capture.isOpened() )
        throw "Error when reading steam_avi";

    namedWindow("Display Image", WINDOW_AUTOSIZE );

    Mat image;

    fpdw::detector::FPDWDetector detector(argv[2], 20);

    sort_tracker tracker;



    for( ; ; )
    {
        capture >> image;

        if(image.empty())
            break;

        resize(image, image,  Size(), 0.75, 0.75, CV_INTER_AREA);
        detector.process(image);
        std::vector<cv::Rect_<float> >detections; //our detections to feed the tracker
        std::vector<cv::Rect> rect = detector.getBBoxes(); //result of detection module
        std::vector<cv::Rect_<float> >tracklets; //our active tracklets

        for(const auto &i : rect)
        {
            cv::rectangle(image, i, cv::Scalar(255, 0, 0), 1);

            detections.push_back(cv::Rect_<float>(i.x,i.y,i.width,i.height));

        }


        tracklets = tracker.update(detections);


        //draw tracklets
        for(const auto t :tracklets)
        {
            cv::rectangle(image, t, cv::Scalar(2, 0, 250), 5);



        }

        imshow("Display Image", image);

        waitKey(1);
    }

    return 0;
}
