/* 
 * File:   CMRPT_Route.h
 * Author: neztol
 *
 * Created on 19 de abril de 2010, 17:59
 */

#include "CRealMatches.h"

typedef struct {
    string timestamp;
    double x;
    double y;
    double z;
    double yaw;
    double pitch;
    double roll;
    string image;
} t_RoutePoint;

#ifndef _CMRPT_ROUTE_H
#define	_CMRPT_ROUTE_H

class CMRPT_Route {
public:
    CMRPT_Route(string path, string file);    
    IplImage * getNearest(t_RoutePoint point_in);
    IplImage * getNext(t_RoutePoint &currentPoint);
    void addPoints(string path, string file);
    CMRPT_Route(const CMRPT_Route& orig);
    virtual ~CMRPT_Route();

private:    

    vector<t_RoutePoint> route;
    int index;

};

#endif	/* _CMRPT_ROUTE_H */

