#include "stdafx.h"
#include "procesaImagen.h"
#include "VLCJavaWrapper.h"
#define _INTERFAZ_LIB_VLC_21_NOV_2006
#include "InterfazLibVLC.h"

map <LPWSTR, manejador_t> imgDispositivos;

t_imagen imagen2IPL(JNIEnv * env, jobject obj, jdouble posX, jdouble posY, 
								jdouble angulo, jint w, jint h, jintArray img, 
								jstring jstr) {
	const char * name;
	jint numChars;
	jsize size;
	jint * imagen;
	boolean isCopy = true;
	t_imagen retorno;
	int ancho, alto;

	name = env->GetStringUTFChars(jstr, 0);
	numChars = env->GetStringUTFLength(jstr);
	size = env->GetArrayLength(img);
	imagen = env->GetIntArrayElements(img, &isCopy);

	retorno.nombre = new char[numChars];
	strcpy(retorno.nombre, name);
	retorno.posX = (double)posX;
	retorno.posY = (double)posY;
	retorno.angulo = (double)angulo;
	ancho = (int)w;
	alto = (int)h;
	
	IplImage * ipl = cvCreateImage(cvSize(ancho, alto), IPL_DEPTH_8U, 3);
	ipl->origin = 1;

	cvZero(ipl);
	for (int i = 0; i < alto; i++) {
		for (int j = 0; j < ancho; j++) {
			unsigned int valor1 = (unsigned int)imagen[i * ancho * 3 + j * 3];
			unsigned int valor2 = (unsigned int)imagen[i * ancho * 3 + j * 3 + 1];
			unsigned int valor3 = (unsigned int)imagen[i * ancho * 3 + j * 3 + 2];

			CV_IMAGE_ELEM(ipl, unsigned char, i, j * 3) = valor1;
			CV_IMAGE_ELEM(ipl, unsigned char, i, j * 3 + 1) = valor2;
			CV_IMAGE_ELEM(ipl, unsigned char, i, j * 3 + 2) = valor3;
		}
	}

	retorno.imagen = ipl;

	env->ReleaseIntArrayElements(img, imagen, 0);
	env->ReleaseStringUTFChars(jstr, name);

	return retorno;
}

IplImage * captura(LPWSTR dispositivo) {
	try {
		map<LPWSTR, manejador_t>::iterator iter = imgDispositivos.find(dispositivo);	
		if (iter == imgDispositivos.end()) {
			int ret = initMC(dispositivo);
			if (ret != EXITO) {
				fprintf(stderr, "Error al obtener la imagen\n");
				return NULL;
			}
		}

		iter = imgDispositivos.find(dispositivo);	
		manejador_t manejador;
		if (iter == imgDispositivos.end()) {
			fprintf(stderr, "El iterador no se actualizo\n");
			return NULL;
		} else {	
			manejador = (manejador_t)iter->second;
		}

		unsigned char * pBuf = manejador.pBuf;
		int ancho = manejador.ancho;
		int alto = manejador.alto;
		int posicion = manejador.posicion;
		int imagesize = manejador.imagesize;
		char * formato = manejador.formato;

		unsigned char * imagen = new unsigned char[imagesize];
		unsigned char * imagen2 = new unsigned char[imagesize];

		CopyMemory(imagen, (unsigned char *)&(pBuf[posicion]), sizeof(unsigned char) * imagesize);

		IplImage * ipl = cvCreateImage(cvSize(ancho, alto), IPL_DEPTH_8U, 3);
		ipl->origin = 1;

		cvZero(ipl);
		for (int i = 0; i < alto; i++) {
			for (int j = 0; j < ancho; j++) {
				unsigned int valor1 = (unsigned int)imagen[i * ancho * 3 + j * 3];
				unsigned int valor2 = (unsigned int)imagen[i * ancho * 3 + j * 3 + 1];
				unsigned int valor3 = (unsigned int)imagen[i * ancho * 3 + j * 3 + 2];

				CV_IMAGE_ELEM(ipl, unsigned char, i, j * 3) = valor1;
				CV_IMAGE_ELEM(ipl, unsigned char, i, j * 3 + 1) = valor2;
				CV_IMAGE_ELEM(ipl, unsigned char, i, j * 3 + 2) = valor3;
			}
		}

		return ipl;
	} catch (char * str) {
		fprintf (stderr, "Excepción al obtener la imagen, %s\n", str);
	}

	return NULL;
}

JNIEXPORT void JNICALL 
Java_ImagenId__1verImagen(JNIEnv * env, jobject obj, jdouble posX, jdouble posY, 
								jdouble angulo, jint ancho, jint alto, jintArray img, 
								jstring nombre) {

	t_imagen imagen = imagen2IPL(env, obj, posX, posY, angulo, ancho, alto, img, nombre);
	
	char texto1[15];
	char texto2[15];
	char texto3[15];

	sprintf(texto1, "PosX: %4.3f", imagen.posX);
	sprintf(texto2, "PosY: %4.3f", imagen.posY);
	sprintf(texto3, "Angulo: %4.3f", imagen.angulo);

	CvFont fuente;
	
	cvInitFont(&fuente, CV_FONT_VECTOR0, 0.4f, 0.4f);
	cvPutText(imagen.imagen, texto1, cvPoint(5, imagen.imagen->height - 15), &fuente, cvScalar(0, 0, 0));
	cvPutText(imagen.imagen, texto2, cvPoint(5, imagen.imagen->height - 30), &fuente, cvScalar(0, 0, 0));
	cvPutText(imagen.imagen, texto3, cvPoint(5, imagen.imagen->height - 45), &fuente, cvScalar(0, 0, 0));

	cvNamedWindow(imagen.nombre, 1);
	cvShowImage(imagen.nombre, imagen.imagen);

	cvWaitKey(0);
}

JNIEXPORT void JNICALL 
Java_CapturaImagen__1verImagen(JNIEnv * env, jobject obj, jstring midisp) {
	const char* cadena = env->GetStringUTFChars(midisp, NULL);
	int len = env->GetStringUTFLength(midisp);
	LPWSTR dispositivo = new wchar_t[len + 1];
	for (int i = 0; i < len; i++) {
		dispositivo[i] = cadena[i];
	}
	dispositivo[len] = L'\0';
	
	IplImage * imagen = captura(dispositivo);

	cvNamedWindow(cadena, 1);
	cvShowImage(cadena, imagen);
	
	cvWaitKey(0);

	cvReleaseImage(&imagen);
}