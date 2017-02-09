#include "camera.h"
int Camera::counter = 0;

Camera::Camera()
{
    id=counter++;
    frame_id = 0;

    if(show_results)
    {
        cv::namedWindow("detections"+std::to_string(id));
        // cv::moveWindow("detections"+std::to_string(id), 100*id, 100*id);
    }
}

Camera::~Camera()
{
    capture.release();
    cv::destroyAllWindows();
}

bool Camera::init(std::string videofile, std::string maskfile, std::string calibfile,fpdw::detector::FPDWDetector * det_,cv::Point2f position)
{
    getMaskfile(maskfile);
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

    m_detections = detect(image,2.3);

    filter_by_mask(mask,m_detections);

    get_world_coordinates(calibMat,m_detections);

    get_confidences(m_detections);

    extract_appearance(m_detections,image);




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


    std::vector<detection> detections;
    for(const auto &i : rect)
    {
        detection new_detection;
        new_detection.bbox = cv::Rect(i.x*factor,i.y*factor,i.width*factor,i.height*factor);
        new_detection.cam_id = id;

        detections.push_back(new_detection);


    }

    return detections;

}

void Camera::filter_by_mask(cv::Mat mask, std::vector<detection> &detections)
{
    for(auto &d : detections)
    {
        cv::Point feet = getFeet(d.bbox);

        d.valid = insideMask(mask,feet);
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

    for(auto &d : detections)
    {
        float confidence = 1.0;

        float r = d.bbox.height/get_distance_to_Camera(d.relative_point);

        if(ratio>10)
        {
            float diff = fabs(r-ratio);

            confidence-=diff/ratio;
        }

        d.confidence = confidence;


    }
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

bool Camera::insideMask(const cv::Mat &mask,cv::Point p)
{
    int pixel = (int)(mask.at<uchar>(p.y,p.x));


    return pixel > 0;
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

    bool uniform = true; bool accumulate = false;

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

void Camera::getMaskfile(std::string maskfile)
{
    mask = cv::imread(maskfile,0);
}

void Camera::getCalibfile(std::string calibfile)
{
    std::fstream myfile(calibfile, std::ios_base::in);

    float a;
    char dummy;
    std::vector<float> calibvector;
    for(int i = 0; i<9; i++)
    {
        myfile>>a;
        myfile>>dummy;
        calibvector.push_back(a);
        std::cout << "reading: " << a << " vector size: " << calibvector.size() << "/9 " << std::endl;
    }


    myfile.close();

    calibMat = cv::Mat( 3,3, CV_32FC1,calibvector.data()).clone();

    std::cout << "Calibration Matrix Initalized:" << calibMat << std::endl;



}
