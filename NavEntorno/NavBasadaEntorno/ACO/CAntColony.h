/* 
 * File:   CAntColony.h
 * Author: neztol
 *
 * Created on 13 de mayo de 2010, 9:15
 */

#ifndef _CANTCOLONY_H
#define	_CANTCOLONY_H

#include "aco.h"

/*int horizonSlider;
int edgeSlider; //75
int bordeSup;

void on_trackbar(int h) {
    if (horizonSlider >= 1)
        bordeSup = horizonSlider;
}//*/

class CAntColony {
public:
    CAntColony(CvSize size);
    CAntColony(const CAntColony& orig);
    virtual ~CAntColony();
    CvPoint * iterate(IplImage * img);
    
private:
    int horizonSlider;
    int edgeSlider; //75
    int bordeSup;
    int kBar, kpBar, kdBar, izqBar, dchaBar;
    int searchAreas;
    int consigna;
    float anguloCamara;
    int errorAnt;

    IplImage* ed;
    IplImage* gray;
    IplImage* dst;
    IplImage* mask;
    IplImage* shadows;

    IplImage* bordes;
    IplImage* shadowMask;
    IplImage* segmented;
    IplImage* traces;
    IplImage* hough;
    IplImage* tracesaux;
    IplImage* tempImage;
    IplImage* tempCapture;

    int keyCode;

    //variables de la colonia

    colonyStruct *colony;
    secuenceStruct shortestLeftPath;
    secuenceStruct shortestRightPath;

    int attractionX, attractionXAnt, attractionY;
    int refLeftY, refRightY;
    int aRef, bRef, cRef, dRef;
    int selectedImage;
    int corte;

    CvMemStorage* storage; //storage y contours: necesarios para la función findcontours
    CvSeq* contour;    
};

#endif	/* _CANTCOLONY_H */

