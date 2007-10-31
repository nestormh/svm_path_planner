#pragma once
#include "bufPolig.h"
#include "Hormigas.h"

class tieneObstaculos
{
private:
	int bSize, params;		// Parámetros del AdaptiveThreshold
	int puntos;				// Nº de puntos del polígono inicial
	bufPolig buf;				// Buffers para calcular el polígono
	int bufSize;			// Tamaño de los buffer
	int orAnd;				// Operacion or/and
	tPoligono puntos1;		// Polígonos anteriores
	int umbral;						// Valor del umbral no adaptativo
		
	Hormigas * hormigas;	// Clase contenedora del algoritmo de las hormigas

	void getContornos(IplImage * img, IplImage * mascara);
	void creaMascaraDesdePoligono(IplImage * img, IplImage * mascara);
	void anadeObjetos(IplImage * imagen, IplImage * mascara);
public:
	tieneObstaculos();
	tieneObstaculos(int bSize, int params, int puntos, int bufSize);	
	IplImage * getMask(IplImage * imagen);

	// Funciones de actualización haciendo uso de los trackBar
	int updateBSize(int val);
	int updateParams(int val);
	int updateBuffer(int val);
	int updatePuntos(int val);
	int updateOrAnd(int val);
	int updateUmbral(int val);
public:
	~tieneObstaculos();
};
