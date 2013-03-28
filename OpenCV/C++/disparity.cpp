#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/contrib/contrib.hpp"
#include <stdio.h>
#include <string.h>
using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
    Mat img1, img2, g1, g2;
    Mat disp, disp8;
    char* method = argv[3];
    img1 = imread(argv[1]);
    img2 = imread(argv[2]);
    cvtColor(img1, g1, CV_BGR2GRAY);
    cvtColor(img2, g2, CV_BGR2GRAY);

    if (!(strcmp(method, "BM")))
    {
        StereoBM sbm;
        sbm.state->SADWindowSize = 9;
        sbm.state->numberOfDisparities = 112;
        sbm.state->preFilterSize = 5;
        sbm.state->preFilterCap = 61;
        sbm.state->minDisparity = -39;
        sbm.state->textureThreshold = 507;
        sbm.state->uniquenessRatio = 0;
        sbm.state->speckleWindowSize = 0;
        sbm.state->speckleRange = 8;
        sbm.state->disp12MaxDiff = 1;
        sbm(g1, g2, disp);
    }
    else if (!(strcmp(method, "SGBM")))
    {
        StereoSGBM sbm;
        sbm.SADWindowSize = 3;
        sbm.numberOfDisparities = 144;
        sbm.preFilterCap = 63;
        sbm.minDisparity = -39;
        sbm.uniquenessRatio = 10;
        sbm.speckleWindowSize = 100;
        sbm.speckleRange = 32;
        sbm.disp12MaxDiff = 1;
        sbm.fullDP = false;
        sbm.P1 = 216;
        sbm.P2 = 864;
        sbm(g1, g2, disp);
    }

    
    normalize(disp, disp8, 0, 255, CV_MINMAX, CV_8U);

    imshow("left", img1);
    imshow("right", img2);
    imshow("disp", disp8);

    waitKey(0);

    return(0);
}
