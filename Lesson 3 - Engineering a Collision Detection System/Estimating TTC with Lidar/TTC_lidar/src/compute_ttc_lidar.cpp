#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>


#include "dataStructures.h"
#include "structIO.hpp"

using namespace std;

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double &TTC)
{
    // auxiliary variables
    double dT = 0.1;        // time between two measurements in seconds
    double laneWidth = 4.0; // assumed width of the ego lane

    // find closest distance to Lidar points within ego lane
    double minXPrev = 1e9, minXCurr = 1e9;
    for (auto point : lidarPointsPrev)
    {   
        // If point is in lane
        if (fabs(point.y) <=laneWidth/2.)
            minXPrev = minXPrev > point.x ? point.x : minXPrev;
    }

    for (auto point : lidarPointsCurr)
    {
        // If point is in lane
        if (fabs(point.y) <=laneWidth/2.)
            minXCurr = minXCurr > point.x ? point.x : minXCurr;
    }

    // compute TTC from both measurements
    TTC = minXCurr * dT / (minXPrev - minXCurr);
}

int main()
{

    std::vector<LidarPoint> currLidarPts, prevLidarPts;
    readLidarPts("../dat/C22A5_currLidarPts.dat", currLidarPts);
    readLidarPts("../dat/C22A5_prevLidarPts.dat", prevLidarPts);


    double ttc;
    computeTTCLidar(prevLidarPts, currLidarPts, ttc);
    cout << "ttc = " << ttc << "s" << endl;
}