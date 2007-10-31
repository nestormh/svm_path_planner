#include "StdAfx.h"
#include "CRuta.h"

CRuta::CRuta(char * path_in, char * file) {
	path = new char[strlen(path_in) + strlen(file) + 1];
	sprintf(path, "%s\\%s", path_in, file);	

	loadRuta(path_in, file);
}

CRuta::~CRuta(void) {	
}

void CRuta::loadRuta(char * path_in, char * file) {
	char * fichero = new char[strlen(path_in) + strlen(file) + 5];

	sprintf(fichero, "%s\\%s.txt", path_in, file);	

	ifstream ifs(fichero, ifstream::in);

	if (ifs.is_open()) {			
		tCoord val;
		while (ifs.good()) {
			ifs >> val.x;			
			ifs >> val.y;			
			ifs >> val.z;			
			ruta.push_back(val);
		}			
		ifs.close();
	} else {
		cout << "Error al abrir el fichero" << endl;
	}	
}

tCoord CRuta::getPosicion(int i) {
	return ruta[i];
}

IplImage * CRuta::getImagenCercana(tCoord pos) {
	int minPos = 0;	
	double minDist = sqrt(pow(ruta[0].x - pos.x, 2) + pow(ruta[0].y - pos.y, 2) + pow(ruta[0].z - pos.z, 2));
	for (int i = 0; i < ruta.size(); i++) {
		double distancia = sqrt(pow(ruta[i].x - pos.x, 2) + pow(ruta[i].y - pos.y, 2) + pow(ruta[i].z - pos.z, 2));
		if (distancia < minDist) {
			minDist = distancia;
			minPos = i;
		}
	}

	char * pref = "Imagen";
	char * post = "a.jpg";
	char * imagen = new char[strlen(path) + strlen(pref) + strlen(post) + 4];

	sprintf(imagen, "%s\\%s%d%s", path, pref, minPos, post);	

	IplImage * retorno = cvLoadImage(imagen);	

	return retorno;
}

IplImage * CRuta::getImagen(int pos) {
	char * pref = "Imagen";
	char * post = "a.jpg";
	char * imagen = new char[strlen(path) + strlen(pref) + strlen(post) + 4];

	sprintf(imagen, "%s\\%s%d%s", path, pref, pos, post);	

	IplImage * retorno = cvLoadImage(imagen);	

	return retorno;
}