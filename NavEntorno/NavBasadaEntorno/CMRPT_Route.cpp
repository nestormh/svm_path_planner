/* 
 * File:   CMRPT_Route.cpp
 * Author: neztol
 * 
 * Created on 19 de abril de 2010, 17:59
 */

#include "CMRPT_Route.h"

CMRPT_Route::CMRPT_Route(string path, string file) {
    addPoints(path, file);
}

void CMRPT_Route::addPoints(string path, string file) {
    string finalPath = path;
    finalPath += file;
    ifstream ifs(finalPath.c_str() , ifstream::in );    
    char line[1024];
    ifs.getline(line, 1024);   

    double val;
    while (ifs.good()) {
        t_RoutePoint point;
        ifs.get(line, 1024, ' ');
        point.timestamp = line;
        if (point.timestamp.size() == 0) break;
        ifs >> point.x;
        ifs >> point.y;
        ifs >> point.z;
        ifs >> point.yaw;
        ifs >> point.pitch;
        ifs >> point.roll;
        point.image = path;
        point.image += "Images_rect/CAMERA_LEFT_";        
        point.image += point.timestamp;
        point.image += ".jpg";
        
        ifs.ignore(1024, '\n');
        
        /*cout << "TS = " << point.timestamp << "; ";
        cout << "x = " << point.x << "; ";
        cout << "y = " << point.y << "; ";
        cout << "z = " << point.z << "; ";
        cout << "yaw = " << point.yaw << "; ";
        cout << "pitch = " << point.pitch << "; ";
        cout << "roll = " << point.roll <<  "; ";
        cout << "image = " << point.image <<  endl;//*/

        route.push_back(point);
    }
    ifs.close();

    index = 0;
}

IplImage * CMRPT_Route::getNearest(t_RoutePoint point_in) {
    double maxAng = 10 * CV_PI / 180;
    double minDist = DBL_MAX;
    vector<t_RoutePoint>::iterator minPos;

    for (vector<t_RoutePoint>::iterator it = route.begin(); it != route.end(); it++) {
        double angDist = max(point_in.yaw, it->yaw) - min(point_in.yaw, it->yaw);
        if (angDist > CV_PI)
            angDist = CV_PI * 2 - angDist;
        if (angDist > maxAng) continue;
        double dist = sqrt(pow(point_in.x - it->x, 2.0) +
                           pow(point_in.y - it->y, 2.0) +
                           pow(point_in.z - it->z, 2.0));
        if (dist > 2.0) continue;
        if (dist < minDist) {
            minDist = dist;
            minPos = it;
        }
    }

    if (minDist == DBL_MAX) {
        //cerr << "No se encontró ninguna imagen cercana" << endl;
        return NULL;
    }

    //cout << "Distancia = " << minDist << endl;

    cout << minPos->image.c_str() << endl;

    return cvLoadImage(minPos->image.c_str(), 0);
}

IplImage * CMRPT_Route::getNext(t_RoutePoint &currentPoint) {
    currentPoint = route.at(index);
    index++;

    if (index == route.size())
        index = 0;

    cout << currentPoint.image.c_str() << endl;
    return cvLoadImage(currentPoint.image.c_str(), 0);
}

void CMRPT_Route::setIndex(int index) {
    this->index = index;
}

CMRPT_Route::CMRPT_Route(const CMRPT_Route& orig) {
}

CMRPT_Route::~CMRPT_Route() {
}

