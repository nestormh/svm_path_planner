#pragma once

typedef struct tCoord {
	double x, y, z, ang;	
} tCoord;

class CRuta {
private:
	char * path;
	vector<tCoord> ruta;
	void loadRuta(char * path_in, char * file);
	double getDifAngulos(double ang1, double ang2);
	int minPos;

	int masde50a;
	int masde100a;
	int masde150a;
	int masde200a;
	int masde250a;
	int masde50b;
	int masde100b;
	int masde150b;
	int masde200b;
	int masde250b;

public:
	CRuta(char * path, char * file);
	~CRuta(void);
	tCoord getPosicion(int i);	
	IplImage * getImagenAt(int i);
	IplImage * getImagenCercana(tCoord pos);
	IplImage * getMask();
	void muestraDistancias();
};
