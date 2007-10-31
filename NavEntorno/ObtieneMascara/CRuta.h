#pragma once

typedef struct tCoord {
	double x, y, z;	
} tCoord;

class CRuta {
private:
	char * path;
	vector<tCoord> ruta;
	void loadRuta(char * path_in, char * file);

public:
	CRuta(char * path, char * file);
	~CRuta(void);
	tCoord getPosicion(int i);	
	IplImage * getImagenCercana(tCoord pos);	
	IplImage * getImagen(int pos);
};
