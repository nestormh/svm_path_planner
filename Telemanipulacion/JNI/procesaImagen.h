// procesaImagen.h: archivo de encabezado principal del archivo DLL de procesaImagen
//

#pragma once

#ifndef __AFXWIN_H__
	#error "incluir 'stdafx.h' antes de incluir este archivo para PCH"
#endif

#include "resource.h"		// Símbolos principales
#include "ImagenId.h"
#include "CapturaImagen.h"
#include <stdio.h>
#include <string.h>
#include "highgui.h"
#include "cv.h"

using namespace std;

typedef struct {
	double posX,
		posY,
		angulo;
	char * nombre;
	IplImage * imagen;
} t_imagen;

t_imagen imagen2IPL(JNIEnv * env, jobject obj, jdouble posX, jdouble posY, 
								jdouble angulo, jint ancho, jint alto, jintArray img, 
								jstring nombre);
IplImage * captura(LPWSTR dispositivo);