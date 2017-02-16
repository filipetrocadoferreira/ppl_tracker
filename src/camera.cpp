#include "camera.h"
int Camera::counter = 0;

Camera::Camera()
{
    id=counter++;
    frame_id = 0;

    if(show_results)
        {
            cv::namedWindow("detections"+std::to_string(id));

        }

    max_x = 9.0;
    max_y = 4.0;
    min_x = 0.5;
    min_y = 0.5;
}

Camera::~Camera()
{
    capture.release();
    cv::destroyAllWindows();

    //reset counter
    counter = 0;
}

bool Camera::init(std::string videofile, std::string calibfile,fpdw::detector::FPDWDetector * det_,cv::Point2f position)
{

    getCalibfile(calibfile);

    capture.open(videofile);
    if(!capture.isOpened())
        return false;


    detector = det_;

    world_position = position;

    return true;
}

bool Camera::process()
{
    m_detections.clear();
    cv::Mat image;
    capture >> image;

    if(image.empty())
        return false;

    m_detections = detect(image,2.3); //resize ratio set to 2.3



    get_world_coordinates(calibMat,m_detections);

    filter_by_local(m_detections);

    get_confidences(m_detections);

    extract_appearance(m_detections,image);


    //show detections (blue : valide , red : invalid)
    if(show_results)
        {
            for(const auto &d : m_detections)
                {
                    cv::rectangle(image, d.bbox, cv::Scalar(255*d.valid, 0, 255*(1-d.valid)), 3);

                    if(d.valid)
                        {
                            cv::putText(image,std::to_string(d.world_point.x)+","+std::to_string(d.world_point.y),d.bbox.tl(),1,3,cv::Scalar(255),2);

                        }


                }
            cv::resize(image, image,  cv::Size(), 0.25, 0.25, CV_INTER_AREA);


            cv::imshow("detections"+std::to_string(id), image);

            cv::waitKey(1);
        }

    frame_id++;
}

std::vector<detection> Camera::detect(const cv::Mat & img, float factor)
{
    cv::Mat image;
    cv::resize(img, image,  cv::Size(), 1/factor, 1/factor, CV_INTER_AREA);

    detector->process(image);
    std::vector<cv::Rect >rect = detector->getBBoxes(); //our detections
    auto confidences = detector->getConfidences();


    std::vector<detection> detections;

    for(int i = 0 ; i < rect.size() ; i++)
        {
            detection new_detection;
            new_detection.bbox = cv::Rect(rect[i].x*factor,rect[i].y*factor,rect[i].width*factor,rect[i].height*factor);
            new_detection.cam_id = id;
            new_detection.confidence = confidences[i];

            detections.push_back(new_detection);


        }

    return detections;

}



void Camera::filter_by_local(std::vector<detection> &detections)
{
    for(auto &d : detections)
        {


            bool valid = true;

            if(d.world_point.x<min_x
                    || d.world_point.y<min_y
                    || d.world_point.x>max_x
                    || d.world_point.y>max_y)
                valid = false;
            d.valid = valid;
    }

}

void Camera::get_world_coordinates(const cv::Mat &calib, std::vector<detection> &detections)
{

    std::vector<cv::Point2f> in;
    std::vector<cv::Point2f> out;
    for(auto &d : detections)
        {
            cv::Point feet = getFeet(d.bbox);

            in.push_back(cv::Point2f((float)feet.x,(float)feet.y));


        }

    if(in.size()>0)
        cv::perspectiveTransform(in,out,calib);

    for(int i = 0 ; i < out.size() ; i++)
        {

            detections[i].world_point = out[i];

            detections[i].relative_point = out[i]-world_position;

            if(get_ratio)
                {
                    if(detections[i].valid)
                        ratio =  get_statistics(detections[i],ratios) ;
                }

        }
}

void Camera::get_confidences( std::vector<detection> &detections)
{

}

void Camera::extract_appearance(std::vector<detection> &detections, cv::Mat &img)
{
    for(auto &d:detections)
        {
            if(d.valid)
                {

                    cv::Mat bbox_img = img(reduce_bbox(d.bbox,0.8,0.5)&cv::Rect(0,0,img.cols,img.rows));

                    d.hist = getHist(bbox_img).clone();
                }
        }
}

float Camera::get_distance_to_Camera(cv::Point2f p)
{
    return (float)cv::norm(p);
}

float Camera::get_statistics( detection d, std::vector<float> &ratios)
{

    float h = d.bbox.height;
    float dist = get_distance_to_Camera(d.relative_point);

    double median;

    ratios.push_back(h/dist);


    size_t size = ratios.size();

    if ( size > 10 )
        {
            std::sort(ratios.begin(), ratios.end());

            if (size  % 2 == 0)
                {
                    median = (ratios[size / 2 - 1] + ratios[size / 2]) / 2;
                }
            else
                {
                    median = ratios[size / 2];
                }
        }

    if (size>1000)
        {
            ratios.erase(ratios.begin());
        }
    return median;

}

cv::Point Camera::getFeet(cv::Rect r)
{
    return cv::Point(r.x+r.width/2,r.y+r.height);
}


cv::Mat Camera::getHist(cv::Mat img)
{
    /// Separate the image in 3 places ( B, G and R )
    std::vector<cv::Mat> bgr_planes;
    cv::split( img, bgr_planes );

    /// Establish the number of bins
    int histSize = 16;

    /// Set the ranges ( for B,G,R) )
    float range[] = { 0, 256 } ;
    const float* histRange = { range };

    bool uniform = true;
    bool accumulate = false;

    cv::Mat b_hist, g_hist, r_hist;

    /// Compute the histograms:
    cv::calcHist( &bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
    cv::calcHist( &bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
    cv::calcHist( &bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );

    /// Normalize the result to [ 0, histImage.rows ]
    cv::normalize(b_hist, b_hist, 0, 255,cv:: NORM_MINMAX, -1, cv::Mat() );
    cv::normalize(g_hist, g_hist, 0, 255,cv:: NORM_MINMAX, -1, cv::Mat() );
    cv::normalize(r_hist, r_hist, 0, 255,cv:: NORM_MINMAX, -1, cv::Mat() );



    cv::Mat hist;
    std::vector<cv::Mat> hists;
    hists.push_back(b_hist);
    hists.push_back(g_hist);
    hists.push_back(r_hist);

    cv::hconcat(hists,hist);


    return hist;

}

cv::Rect Camera::reduce_bbox(cv::Rect r, float h_r, float v_r)
{
    int cx = r.x+r.width/2;
    int cy = r.y+r.height/2;

    int w = r.height*h_r;
    int h = r.width*v_r;

    return cv::Rect(cx-w/2,cy-h/2,w,h);
}



void Camera::getCalibfile(std::string calibfile)
{
    std::fstream myfile(calibfile, std::ios_base::in);

    float a;
    char dummy; //receives the ','
    std::vector<float> calibvector;
    for(int i = 0; i<9; i++)
        {
            myfile>>a;
            myfile>>dummy;
            calibvector.push_back(a);

        }


    myfile.close();

    calibMat = cv::Mat( 3,3, CV_32FC1,calibvector.data()).clone();

    std::cout << "Calibration Matrix Initialized:" << calibMat << std::endl;



}
