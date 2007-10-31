#include <windows.h>
#include <stdlib.h>
#include <map>
#include "highgui.h"
#include "cv.h"
#include "dshow.h"

using namespace std;

#define EXITO				0
#define NO_LIB				-1
#define NO_ENCONTRADO		-2
#define FUNCION_FALLA		-3

typedef struct manejador_t {
	LPWSTR devicename;
	HANDLE hMapFile;
	unsigned char * pBuf;
	int ancho;
	int alto;	
	int posicion;
	int imagesize;
	char * formato;
} manejador_t;

class CCapturaVLC {
private:
	map <LPWSTR, manejador_t> imgDispositivos;
public:
	LPWSTR * listaDispositivos(int * tamanoLista);
	int initMC(LPWSTR dispositivo);
	int endMC();
	int endMC(LPWSTR dispositivo);
	IplImage * captura(LPWSTR dispositivo);
};