/* Lucas Kanade Tracker using Optical Flow and Forward Backward Error correction.
 * Author: Jay Rambhia
 * Website: http://jayrambhia.com/
 * Blog: http://jayrambhia.com/blog/
 */

/* Include the required libraries */
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/video/video.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<stdio.h>

using namespace cv;
using namespace std;

/* Declare global variables */
Point point1, point2; /* vertical points of the bounding box */
int drag = 0;
Rect rect; /* bounding box */
Mat img, roiImg; /* roiImg - the part of the image in the bounding box */
int select_flag = 0;

/* Declare functions */
void predictBB(vector<Point2f> old_pts, vector<Point2f> new_pts);
void mouseHandler(int event, int x, int y, int flags, void* param);

int main()
{
    int k, i;
    VideoCapture cap = VideoCapture(0);
    cap >> img;
    imshow("image", img);
    while(1)
    {
        cap >> img;
        cvSetMouseCallback("image", mouseHandler, NULL); /* MouseCallBack to select the bounding box */
        if (select_flag == 1)
        {
            vector<Point2f> old_features, new_features, final_features, target_features, old_pts, new_pts;
            vector<uchar> status1, status2;
            vector<float> err1, err2;
            TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03);
            while(1)
            {
                Mat g, newg, oldg;
                
                /* goodFeaturesToTrack and OpticalFlow require grascale iamges */
                cvtColor(img, oldg, CV_BGR2GRAY);
                cvtColor(roiImg, g, CV_BGR2GRAY);
                
                /* find good features to track */
                goodFeaturesToTrack(g, old_features, 6000, 0.6, 2, Mat(), 2);   
                for (i=0; i < old_features.size(); i++)
                {
                    /* Features are tracked from the cropped image.
                     * Hence add point1 x and y to get the correspoding features in
                     * the original imgae. */
                    old_features[i].x = old_features[i].x+point1.x;
                    old_features[i].y = old_features[i].y+point1.y;
                }
                cap >> img;
                cvtColor(img, newg, CV_BGR2GRAY);
                
                /* calcOpticalFlowPyrLK with Forward-Backward Error */
                calcOpticalFlowPyrLK(oldg, newg, old_features, new_features, status1, err1, Size(10, 10), 3, termcrit, 0, 0.001);
                calcOpticalFlowPyrLK(newg, oldg, new_features, final_features, status2, err2, Size(10, 10), 3, termcrit, 0, 0.001);
                for (i=0; i < new_features.size(); i++)
                {
                    if (status1[i] && status2[i])
                    {
                        target_features.push_back(final_features[i]);
                    }
                }
                
                for (int j=0; j < target_features.size(); j++)
                {
                    /* Draw tracked points */
                    circle(img, target_features[j], 4, Scalar(255, 0, 0), -1, 8, 0);
                }
                
                new_pts = target_features;
                
                /* Predict new Bounding Box */
                predictBB(old_pts, new_pts);
                old_pts = new_pts;
                
                /* show the image with predicted BB and tracked points */
                imshow("image", img);
                k = waitKey(30);
                if (k == 27)
                {
                    break;
                }
                
                /* Clear all the features and errors */
                final_features.clear();
                old_features.clear();
                new_features.clear();
                target_features.clear();
                status1.clear();
                status2.clear();
                err1.clear();
                err2.clear();
            }
            return 0;
        }
        imshow("image", img);
        k = waitKey(30);
        if (k == 27)
        {
            break;
        }
    }
    return 0;
}

void mouseHandler(int event, int x, int y, int flags, void* param)
{
    if (event == CV_EVENT_LBUTTONDOWN && !drag)
    {
        /* left button clicked. ROI selection begins */
        point1 = Point(x, y);
        drag = 1;
    }
    
    if (event == CV_EVENT_MOUSEMOVE && drag)
    {
        /* mouse dragged. ROI being selected */
        Mat img1 = img.clone();
        point2 = Point(x, y);
        rectangle(img1, point1, point2, CV_RGB(255, 0, 0), 3, 8, 0);
        imshow("image", img1);
    }
    
    if (event == CV_EVENT_LBUTTONUP && drag)
    {
        point2 = Point(x, y);
        rect = Rect(point1.x,point1.y,x-point1.x,y-point1.y);
        drag = 0;
        roiImg = img(rect);
    }
    
    if (event == CV_EVENT_LBUTTONUP)
    {
        select_flag = 1;
        drag = 0;
    }
}

/* Function to predict the new bounding box
 * WARNING: This function sucks. I am not changing the size of the bounding box at all
 * and also prediction is very bad.
 * Need to come up with something better
 */
void predictBB(vector<Point2f> old_pts, vector<Point2f> new_pts)
{
    int pts, i, cen_dx, cen_dy, cen_x, cen_y;
    vector<float> dx, dy;
    if (old_pts.size() == 0)
    {
        /* Initially old_pts is empty */
        old_pts = new_pts;
    }
    if (old_pts.size() < new_pts.size())
    {
        pts = old_pts.size();
    }
    else
    {
        pts = new_pts.size();
    }
    for (i=0; i<pts; i++)
    {
        /* get the difference between new and old points */
        dx.push_back(new_pts[i].x - old_pts[i].x);
        dy.push_back(new_pts[i].y - old_pts[i].y);
    }
    
    if (dx.size() == 0 || dy.size() == 0)
    {
        return;
    }
    
    /* Predict new center of the bounding box */
    for (i=0, cen_dx=0, cen_dy=0; i<dx.size(); i++)
    {
        cen_dx += int(dx[i]);
        cen_dy += int(dy[i]);
    }
    cen_x = (cen_dx/pts)/3;
    cen_y = (cen_dy/pts)/3;
    
    /* Draw new bounding box */
    point1 = Point(point1.x+cen_x, point1.y+cen_y);
    point2 = Point(point2.x+cen_x, point2.y+cen_y);
    if (point1.x == point2.x || point1.y == point2.y)
    {
        return;
    }
    rect = Rect(point1.x, point1.y, point2.x - point1.x, point2.y - point1.x);
    rectangle(img, point1, point2, CV_RGB(255, 0, 0), 3, 8, 0);
}
