/* Tracker based on FLANN matching of SIFT keypoints
 * Author: Jay Rambhia
 * Website: http://jayrambhia.com/
 * Blog: http://jayrambhia.com/blog/
 */

#include<opencv2/opencv.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/imgproc/imgproc_c.h>
#include<opencv2/nonfree/nonfree.hpp>
#include<stdio.h>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
using namespace cv;
using namespace std;

/* Declare global variables */
Point point1, point2; /* vertical points of the bounding box */
int drag = 0;
Rect rect; /* bounding box */
Mat img, roiImg; /* roiImg - the part of the image in the bounding box */
int select_flag = 0;

/* Declare functions */
void mouseHandler(int event, int x, int y, int flags, void* param);
void predictBB(vector<KeyPoint> keypoints_roi, vector<KeyPoint> keypoints_img);
double getRadius(Point cen, vector<KeyPoint> keypoints);
Point getCenter(vector<KeyPoint> keypoints);
vector<double> getDiff(Point cen, double rad);
void newPoints(vector<double> diff, Point cen, double rad);

int main()
{
    int k, i;
    VideoCapture cap = VideoCapture(0);
    cap >> img;
    imshow("image", img);
    SIFT sift;                                      /* SIFT */
    vector<KeyPoint> keypoints_roi, keypoints_img;  /* keypoints found using SIFT */
    Mat descriptor_roi, descriptor_img;             /* Descriptors for SIFT */
    FlannBasedMatcher matcher;                      /* FLANN based matcher to match keypoints */
    vector<DMatch> matches, good_matches;           /* DMatch used to match keypoints */
    
    while(1)
    {
        cap >> img;
        cvSetMouseCallback("image", mouseHandler, NULL); /* MouseCallBack to select the bounding box */
        if (select_flag == 1)
        {
            sift(roiImg, Mat(), keypoints_roi, descriptor_roi);      /* get keypoints of ROI image */
            sift(img, Mat(), keypoints_img, descriptor_img);         /* get keypoints of the image */
            matcher.match(descriptor_roi, descriptor_img, matches);  /* Match keypoints using FLANN */
            
            /* Filter matched keypoints depending upon the distance */             
            for (i=0; i < descriptor_roi.rows; i++)
            {
                if (matches[i].distance < 0.5)
                {
                    good_matches.push_back(matches[i]);
                }
            }
            
            /* Draw matched keypoints */
            Mat img_matches;
            drawMatches(roiImg, keypoints_roi, img, keypoints_img, 
                        good_matches, img_matches, Scalar::all(-1), 
                        Scalar::all(-1), vector<char>(), 
                        DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

            /* get matched keypoints using DMatch */
            vector<KeyPoint> keypoints1, keypoints2; 
            for (i=0; i<good_matches.size(); i++)
            {
                keypoints1.push_back(keypoints_img[good_matches[i].trainIdx]);
            }
            for (i=0; i<good_matches.size(); i++)
            {
                keypoints2.push_back(keypoints_roi[good_matches[i].queryIdx]);
            }
            /* predict new ROI */
            predictBB(keypoints2, keypoints1);
            
            /* Draw matched keypoints on ROI image, and the origianl image */
            Mat img_matches1, roi_matches1;
            drawKeypoints(img, keypoints1, img_matches1, Scalar::all(-1), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            drawKeypoints(roiImg, keypoints2, roi_matches1, Scalar::all(-1), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            
            /* show all the images */
            imshow("img_matches", img_matches);
            imshow("img_matches1", img_matches1);
            imshow("roi_matches1", roi_matches1);
            imshow("SIFT Tracker", img);
            k = waitKey(30);
            if (k == 27)
            {
                break;
            }
            
            /* clear all the features */
            keypoints_roi.clear();
            keypoints_img.clear();
            matches.clear();
            good_matches.clear();
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
    if (event == CV_EVENT_LBUTTONDOWN && !drag && !select_flag)
    {
        /* left button clicked. ROI selection begins */
        point1 = Point(x, y);
        drag = 1;
    }
    
    if (event == CV_EVENT_MOUSEMOVE && drag && !select_flag)
    {
        /* mouse dragged. ROI being selected */
        Mat img1 = img.clone();
        point2 = Point(x, y);
        rectangle(img1, point1, point2, CV_RGB(255, 0, 0), 3, 8, 0);
        imshow("image", img1);
    }
    
    if (event == CV_EVENT_LBUTTONUP && drag && !select_flag)
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
 * WARNING: This function sucks. prediction is very bad.
 * Need to come up with something better
 */
 
void predictBB(vector<KeyPoint> keypoints_roi, vector<KeyPoint> keypoints_img)
{
    int i;
    double rad_old, rad_new;
    Point center_old, center_new;
    vector<double> diff;
    
    /* KeyPoints are obtained from the cropped image.
     * Hence add point1 x and y to get the correspoding features in
     * the original imgae. */
    for (i=0; i < keypoints_roi.size(); i++)
    {
        keypoints_roi[i].pt.x += point1.x;
        keypoints_roi[i].pt.y += point1.y;
    }
    drawKeypoints(img, keypoints_roi, img, Scalar::all(-1), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    
    /* get the center(mean) of all the matched keypoints
     * see getCenter(vector<KeyPoint> keypoints) */
    center_old = getCenter(keypoints_roi);
    center_new = getCenter(keypoints_img);
    
    /* get the radius of the circle
     * see getRadius(Point cen, vector<KeyPoint> keypoints) */
    rad_old = getRadius(center_old, keypoints_roi);
    rad_new = getRadius(center_new, keypoints_img);
    
    /* get the difference
     * see getDiff(Point cen, double rad) */
    diff = getDiff(center_old, rad_old);
    
    /* get points of the new bounding box */
    newPoints(diff, center_new, rad_new);
    
    /* Draw new ROI */
    rect = Rect(point1.x, point1.y, point2.x - point1.x, point2.y - point1.y);
    rectangle(img, point1, point2, CV_RGB(255, 0, 0), 3, 8, 0);
    roiImg = img(rect);
}

/* get Center of all the points. i.e. mean co-ordinate of all the keypoints */
Point getCenter(vector<KeyPoint> keypoints) 
{
    int i;
    int cen_x = 0, cen_y = 0;
    Point cen;
    for (i=0; i<keypoints.size(); i++)
    {
        cen_x += keypoints[i].pt.x;
        cen_y += keypoints[i].pt.y;
    }
    cen.x = cen_x/keypoints.size();
    cen.y = cen_y/keypoints.size();
    return cen;
}

/* get Radius of a circle which encloses all the matched keypoints.
 * The center of the circle is the mean of all the matched keypoints.
 * see getCenter(vector<KeyPoint> keypoints)
 */
double getRadius(Point cen, vector<KeyPoint> keypoints)
{
    int i;
    double rad=0, dis_euc;
    for (i=0; i < keypoints.size(); i++)
    {
        dis_euc = sqrt((cen.x-keypoints[i].pt.x)*(cen.x-keypoints[i].pt.x)+(cen.y-keypoints[i].pt.y)*(cen.y-keypoints[i].pt.y));
        if (rad < dis_euc)
        {
            rad = dis_euc;
        }
    }
    return rad;
}

/* get Difference between the bounding box and the circle
 * see getRadius(Point cen, vector<KeyPoint> keypoints)
 */
vector<double> getDiff(Point cen, double rad)
{
    vector<double> diff;
    diff.push_back(cen.x - rad - point1.x);
    diff.push_back(point2.x - (cen.x + rad));
    diff.push_back(cen.y - rad - point1.y);
    diff.push_back(point2.y - (cen.y + rad));
    
    return diff;
}

/* get new points of the bounding box */
void newPoints(vector<double> diff, Point cen, double rad)
{
    printf("rad = %lf\tcen=(%d, %d)\n",rad, cen.x, cen.y);
    printf("%f %f %f %f\n",diff[0], diff[1], diff[2], diff[3]);
    point1.x = cen.x - rad - diff[0];
    point1.y = cen.y - rad - diff[1];
    point2.x = cen.x + rad + diff[2];
    point2.y = cen.y + rad + diff[3];
    printf("(%d, %d), (%d, %d)\n", point1.x, point1.y, point2.x, point2.y);
}
