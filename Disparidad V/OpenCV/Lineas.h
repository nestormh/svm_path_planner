#include <cv.h>
#include <highgui.h>
#include <stdlib.h>
#include <stdio.h>

#ifndef LINEAS_H_
#define LINEAS_H_

class Lineas {
	
private:
	int n;							// No. de elementos insertados en posiciones distintas
	int max;						// No. máximo de listas de líneas
	int *index;						// Vector almacena las posiciones ocupadas
	CvSeq **lines;					// Líneas
	CvMemStorage* storage;			// Almacenamiento para las CvSeq
	
protected:
	static int Compare (const void * a, const void * b);


	
public:
	Lineas();
	Lineas(int size);
	virtual ~Lineas();
	void Insert (CvPoint *item, int pos);
	void Sort ();
	CvSeq *GetLine(int pos);
	int GetN();
	int *GetIndex();
	void Print();
};



#endif /*LINEAS_H_*/

