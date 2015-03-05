/*
 *  Copyright 2013 Néstor Morales Hernández <nestor@isaatc.ull.es>
 * 
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 * 
 *      http://www.apache.org/licenses/LICENSE-2.0
 * 
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <iostream>

#include <opencv2/opencv.hpp>

using namespace std;

int main() {
    
    cv::Mat inflatedMap = cv::imread("/tmp/inflatedMap.bmp", 0);
    
    vector<vector<cv::Point> > contours;
    cv::findContours(inflatedMap, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

    inflatedMap = cv::Mat::zeros(cv::Size(inflatedMap.cols, inflatedMap.rows), CV_8UC1);
    for (vector<vector<cv::Point> >::iterator it = contours.begin(); it != contours.end(); it++) {
        for (vector<cv::Point>::iterator it2 = it->begin(); it2 != it->end(); it2++) {
            cv::Point2i point(it2->x, it2->y);
            
            inflatedMap.at<char>(it2->y, it2->x) = 255;
        }
    }
    
    
    cv::imshow("inflatedMap", inflatedMap);
    
    cv::waitKey(0);
    
    return 0;
}