#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "detector/fpdw_detector.h"
#include "sort.h"
#include "camera.h"

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


    fpdw::detector::FPDWDetector *ptr_detector;
    ptr_detector = new fpdw::detector::FPDWDetector(argv[2], 15);

    std::vector<Camera> camNetwork;



        std::string videofile0    =  "/home/filipeferreira/workspace/ppl_tracker/Files/Arc/2-actor/distinct-colors/arc_2p_dc_co_C10.mp4";
    //std::string videofile0    =  "/home/filipeferreira/workspace/ppl_tracker/Files/Arc/5-actor/equal-colors/arc_mp_ec_C10.mp4";
    std::string maskfile0     =  "/home/filipeferreira/workspace/ppl_tracker/calibration/0.bmp";
    std::string calibfile0    =  "/home/filipeferreira/workspace/ppl_tracker/calibration/0.txt";


    Camera camera0;
    camera0.init(videofile0,maskfile0,calibfile0,ptr_detector,cv::Point2f(8,4.25));
    camNetwork.push_back(camera0);


     std::string videofile1    =  "/home/filipeferreira/workspace/ppl_tracker/Files/Arc/2-actor/distinct-colors/arc_2p_dc_co_C11.mp4";
    //std::string videofile1    =  "/home/filipeferreira/workspace/ppl_tracker/Files/Arc/5-actor/equal-colors/arc_mp_ec_C11.mp4";
    std::string maskfile1     =  "/home/filipeferreira/workspace/ppl_tracker/calibration/1.bmp";
    std::string calibfile1    =  "/home/filipeferreira/workspace/ppl_tracker/calibration/1.txt";

    Camera camera1;
    camera1.init(videofile1,maskfile1,calibfile1,ptr_detector,cv::Point2f(8,0));
    camNetwork.push_back(camera1);

    std::string videofile2    =  "/home/filipeferreira/workspace/ppl_tracker/Files/Arc/2-actor/distinct-colors/arc_2p_dc_co_C12.mp4";
    //std::string videofile2    =  "/home/filipeferreira/workspace/ppl_tracker/Files/Arc/5-actor/equal-colors/arc_mp_ec_C12.mp4";
    std::string maskfile2     =  "/home/filipeferreira/workspace/ppl_tracker/calibration/2.bmp";
    std::string calibfile2    =  "/home/filipeferreira/workspace/ppl_tracker/calibration/2.txt";

    Camera camera2;
    camera2.init(videofile2,maskfile2,calibfile2,ptr_detector,cv::Point2f(0,0));
    camNetwork.push_back(camera2);

    std::string videofile3    =  "/home/filipeferreira/workspace/ppl_tracker/Files/Arc/2-actor/distinct-colors/arc_2p_dc_co_C13.mp4";
    //std::string videofile3    =  "/home/filipeferreira/workspace/ppl_tracker/Files/Arc/5-actor/equal-colors/arc_mp_ec_C13.mp4";
    std::string maskfile3     =  "/home/filipeferreira/workspace/ppl_tracker/calibration/3.bmp";
    std::string calibfile3    =  "/home/filipeferreira/workspace/ppl_tracker/calibration/3.txt";

    Camera camera3;
    camera3.init(videofile3,maskfile3,calibfile3,ptr_detector,cv::Point2f(0,4.25));
    camNetwork.push_back(camera3);



    std::vector<entry_conditions> entry;
    std::vector<entry_conditions> leave;

    entry_conditions c1;
    c1.cam_id = 1;
    c1.point  =  cv::Point2f (4,4.25);
    c1.dist   = 0.9;

    entry_conditions c2;
    c2.cam_id =  2;
    c2.point  =  cv::Point2f (1,3.25);
    c2.dist   =  0.5;

    entry.push_back(c1);
    entry.push_back(c2);

    leave.push_back(c2);
    leave.push_back(c1);

    sort_tracker tracker(entry,leave);



    bool control = true;

    while(control)
    {
        auto t0 = cv::getTickCount();
        std::vector<std::vector<detection>> net_detections;

        for(auto &cam : camNetwork)
        {
            control = cam.process();


            net_detections.push_back(cam.get_detections());
        }


        std::cout << " GOT: " << net_detections.size() << std::endl;
        tracker.update(net_detections);


        cv::Mat result = tracker.draw_state();


        auto t1 = cv::getTickCount();
        auto time = (t1-t0)/cv::getTickFrequency();

        std::cout << " Time: " << time*1000 << " ms " << std::endl;

        cv::imshow("result",result);
        cv::waitKey(1);

    }






    return 0;
}
