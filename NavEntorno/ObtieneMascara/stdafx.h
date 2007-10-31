// stdafx.h: archivo de inclusión de los archivos de inclusión estándar del sistema
// o archivos de inclusión específicos de un proyecto utilizados frecuentemente,
// pero rara vez modificados
//

#pragma once


#define WIN32_LEAN_AND_MEAN		// Excluir material rara vez utilizado de encabezados de Windows
#include <stdio.h>
#include <tchar.h>
#include <iostream>
#include <time.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <winsock2.h>

#include "highgui.h"
#include "cv.h"

using namespace std;

typedef struct tPoligono {
	CvPoint p1,    // Inferior izquierda
		p2,			// Inferior derecha
		p3,			// Superior izquierda
		p4,			// Superior derecha
		p5;			// Tope superior
} tPoligono;



// TODO: mencionar aquí los encabezados adicionales que el programa necesita
