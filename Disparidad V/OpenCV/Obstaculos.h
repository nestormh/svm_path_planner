/*
 * Obstaculos.h
 *
 *  Created on: 09/11/2009
 *      Author: jonatan
 */
#include <cv.h>
#include <highgui.h>
#include <stdlib.h>
#include <stdio.h>

#ifndef OBSTACULOS_H_
#define OBSTACULOS_H_

class Obstaculos;

typedef struct obstaculo{			// Tipo de datos para definir un obstáculo
			int delta,
				u,
				v,
				width,
				height;

			obstaculo *forward,		// Puntero hacia adelante (frame siguiente)
						*backward;	// Puntero hacia atrás (frame anterior)

			bool discard,
				 added;
		} obstaculo;


class Obstaculos {


private:
	int n;
	int frame;

	CvSeq *list;					// Lista de obstáculos
	CvMemStorage* storage;			// Almacenamiento para las CvSeq

protected:
	static int Compare (const void * a, const void * b, void *userdata);

public:
	Obstaculos();
	Obstaculos(int frame);
	int getN ();
	int getDelta();
	void Insert(int delta, int u, int v, int width, int height);
	void Insert(int delta, int u, int v, int width, int height, bool discard, bool added);
	void Print();
	void Draw(IplImage* src);
	obstaculo* getObstacle(int index);
	void Unlink();
	void Sort (int order);
	void CutBackwards();
	void Save(FILE *filename);
	static int Area(obstaculo *o1, obstaculo *o2);
	virtual ~Obstaculos();
};

#endif /* OBSTACULOS_H_ */
