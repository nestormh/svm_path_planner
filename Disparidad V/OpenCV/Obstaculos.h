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

class Obstaculos {

typedef struct {			// Tipo de datos para definir un obstáculo
		int delta,
			u,
			v,
			width,
			height;
	} obstaculo;

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
	void Insert(int delta, int u, int v, int width, int height);
	void Print();
	void Draw(IplImage* src);
	void getObstacle(int index, obstaculo *returned);
	void Unlink();
	void Sort (int order);
	void Save(FILE *filename);
	virtual ~Obstaculos();
};

#endif /* OBSTACULOS_H_ */
