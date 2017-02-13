#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "detector/fpdw_detector.h"
#include "sort.h"
#include "camera.h"
#include <cstdlib>

using namespace cv;

void init_test( std::vector<Camera> &camNetwork,std::vector<entry_conditions> & entry, std::vector<entry_conditions> & leave, fpdw::detector::FPDWDetector *ptr_detector, int test_number);

int main(int argc, char** argv )
{
    if ( argc != 2 )
        {
            std::cout << argc << std::endl;
            printf("./tracker <test_number>\n"
                   "number between 1 and 5. OR -1 to test all the files\n");
            return -1;
        }

    //Receive the tests to do, if they're valid
    std::vector<int> tests;
    int test = atoi(argv[1]);
    if(test>0 && test<=5)
        {
            tests.push_back(test);
        }
    else if(test == -1)
        {
            std::cout << " Doing all the files" << std::endl;
            //do all the tests
            for (int i=0 ; i < 5 ; i++)
                {
                    tests.push_back(i+1);

                }
        }
    else
        {
            std::cout << "Invalid test number. Please chose between 1 and 5, or -1 to do all the tests" << std::endl;
            return -1;
        }



    //init our detector
    fpdw::detector::FPDWDetector *ptr_detector;
    ptr_detector = new fpdw::detector::FPDWDetector("../detector/inria_detector.xml", 25);

    for(auto t : tests)
        {
            std::vector<Camera> camNetwork;

            std::vector<entry_conditions> entry;
            std::vector<entry_conditions> leave;


            init_test(camNetwork,entry,leave,ptr_detector,t);

            sort_tracker tracker(entry,leave);

            //create logger file

            ofstream outputFile("../results/"+std::to_string(t)+".txt");


            bool control = true;
            int frame_count = 0;

            while(control)
                {
                    auto t0 = cv::getTickCount();
                    std::vector<std::vector<detection>> net_detections;

                    for(auto &cam : camNetwork)
                        {
                            control = cam.process();


                            net_detections.push_back(cam.get_detections());
                        }



                    auto results = tracker.update(net_detections);


                    cv::Mat result = tracker.draw_state();

                    //log results
                    for(auto &r : results)
                        {
                            outputFile<<frame_count<<","<<r.id<<","<<r.world_position<<std::endl;
                        }


                    frame_count++;

                    auto t1 = cv::getTickCount();
                    auto time = (t1-t0)/cv::getTickFrequency();

                    std::cout << " Time: " << time*1000 << " ms " << std::endl;

                    cv::imshow("result",result);
                    cv::waitKey(1);

                }


            outputFile.close();

        }
    cv::destroyAllWindows();


    return 0;
}

void init_test(std::vector<Camera> &camNetwork, std::vector<entry_conditions> &entry, std::vector<entry_conditions> &leave, fpdw::detector::FPDWDetector *ptr_detector, int test_number)
{
    //strings for camera 0
    std::string maskfile0     =  "../calibration/0.bmp";
    std::string calibfile0    =  "../calibration/0.txt";
    //strings for camera 1
    std::string maskfile1     =  "../calibration/1.bmp";
    std::string calibfile1    =  "../calibration/1.txt";
    //strings for camera 2
    std::string maskfile2     =  "../calibration/2.bmp";
    std::string calibfile2    =  "../calibration/2.txt";
    //strings for camera 3
    std::string maskfile3     =  "../calibration/3.bmp";
    std::string calibfile3    =  "../calibration/3.txt";


    std::string videofile0;
    std::string videofile1;
    std::string videofile2;
    std::string videofile3;

    Camera camera0;
    Camera camera1;
    Camera camera2;
    Camera camera3;

    entry_conditions c1;
    entry_conditions c2;


    std::cout << " init with test: " << test_number << std::endl;
    switch (test_number)
        {
        case 1:

            videofile0    =  "../Files/Triangle/1-actor/1-camera/pos_tri_1p_C10.mp4";

            camera0.init(videofile0,maskfile0,calibfile0,ptr_detector,cv::Point2f(8,4.25));
            camNetwork.push_back(camera0);



            c1.cam_id = 0;
            c1.point  =  cv::Point2f (5,2.25);
            c1.dist   = 0.9;

            entry.push_back(c1);



            break;

        case 2:

            videofile0    =  "../Files/Square/1-actor/2-cameras/pos_sqr_1p_C10.mp4";
            videofile1    =  "../Files/Square/1-actor/2-cameras/pos_sqr_1p_C11.mp4";

            camera0.init(videofile0,maskfile0,calibfile0,ptr_detector,cv::Point2f(8,4.25));
            camNetwork.push_back(camera0);

            camera1.init(videofile1,maskfile1,calibfile1,ptr_detector,cv::Point2f(8,0));
            camNetwork.push_back(camera1);
            c1.cam_id = 0;
            c1.point  =  cv::Point2f (5,2.25);
            c1.dist   = 0.9;

            entry.push_back(c1);

            break;

        case 3:

            videofile0    =  "../Files/Arc/1-actor/arc_1p_C10.mp4";
            videofile1    =  "../Files/Arc/1-actor/arc_1p_C11.mp4";
            videofile2    =  "../Files/Arc/1-actor/arc_1p_C12.mp4";
            videofile3    =  "../Files/Arc/1-actor/arc_1p_C13.mp4";


            camera0.init(videofile0,maskfile0,calibfile0,ptr_detector,cv::Point2f(8,4.25));
            camNetwork.push_back(camera0);


            camera1.init(videofile1,maskfile1,calibfile1,ptr_detector,cv::Point2f(8,0));
            camNetwork.push_back(camera1);


            camera2.init(videofile2,maskfile2,calibfile2,ptr_detector,cv::Point2f(0,0));
            camNetwork.push_back(camera2);


            camera3.init(videofile3,maskfile3,calibfile3,ptr_detector,cv::Point2f(0,4.25));
            camNetwork.push_back(camera3);


            c1.cam_id = 1;
            c1.point  =  cv::Point2f (4,4.25);
            c1.dist   = 0.9;

            c2.cam_id =  2;
            c2.point  =  cv::Point2f (1,4.25);
            c2.dist   =  0.5;

            entry.push_back(c1);
            leave.push_back(c2);

            break;

        case 4:

            videofile0    =  "../Files/Arc/2-actor/distinct-colors/arc_2p_dc_co_C10.mp4";
            videofile1    =  "../Files/Arc/2-actor/distinct-colors/arc_2p_dc_co_C11.mp4";
            videofile2    =  "../Files/Arc/2-actor/distinct-colors/arc_2p_dc_co_C12.mp4";
            videofile3    =  "../Files/Arc/2-actor/distinct-colors/arc_2p_dc_co_C13.mp4";


            camera0.init(videofile0,maskfile0,calibfile0,ptr_detector,cv::Point2f(8,4.25));
            camNetwork.push_back(camera0);

            camera1.init(videofile1,maskfile1,calibfile1,ptr_detector,cv::Point2f(8,0));
            camNetwork.push_back(camera1);


            camera2.init(videofile2,maskfile2,calibfile2,ptr_detector,cv::Point2f(0,0));
            camNetwork.push_back(camera2);


            camera3.init(videofile3,maskfile3,calibfile3,ptr_detector,cv::Point2f(0,4.25));
            camNetwork.push_back(camera3);


            c1.cam_id = 1;
            c1.point  =  cv::Point2f (4,4.25);
            c1.dist   = 0.9;


            c2.cam_id =  2;
            c2.point  =  cv::Point2f (1,3.25);
            c2.dist   =  0.5;

            entry.push_back(c1);
            entry.push_back(c2);

            leave.push_back(c2);
            leave.push_back(c1);


            break;

        case 5:

            videofile0    =  "../Files/Arc/5-actor/equal-colors/arc_mp_ec_C10.mp4";
            videofile1    =  "../Files/Arc/5-actor/equal-colors/arc_mp_ec_C11.mp4";
            videofile2    =  "../Files/Arc/5-actor/equal-colors/arc_mp_ec_C12.mp4";
            videofile3    =  "../Files/Arc/5-actor/equal-colors/arc_mp_ec_C13.mp4";


            camera0.init(videofile0,maskfile0,calibfile0,ptr_detector,cv::Point2f(8,4.25));
            camNetwork.push_back(camera0);


            camera1.init(videofile1,maskfile1,calibfile1,ptr_detector,cv::Point2f(8,0));
            camNetwork.push_back(camera1);


            camera2.init(videofile2,maskfile2,calibfile2,ptr_detector,cv::Point2f(0,0));
            camNetwork.push_back(camera2);


            camera3.init(videofile3,maskfile3,calibfile3,ptr_detector,cv::Point2f(0,4.25));
            camNetwork.push_back(camera3);


            c1.cam_id = 1;
            c1.point  =  cv::Point2f (4,4.25);
            c1.dist   = 0.9;

            c2.cam_id =  2;
            c2.point  =  cv::Point2f (1,3.25);
            c2.dist   =  0.5;

            entry.push_back(c1);
            leave.push_back(c2);


            break;
        default: //let's do the seven
            break;
        }
}
