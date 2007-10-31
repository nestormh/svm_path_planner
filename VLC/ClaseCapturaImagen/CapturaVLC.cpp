#include "stdafx.h"
#include "CapturaVLC.h"

LPWSTR * CCapturaVLC::listaDispositivos(int * tamanoLista) {
	// Variables necesarias
	HRESULT hr = 0;
	ICaptureGraphBuilder2 *pCaptureGraph = NULL;	
	IBaseFilter *pVideoInputFilter = NULL;
	ICreateDevEnum *pSysDevEnum = NULL;
	IEnumMoniker *pEnumCat = NULL;	
	LPWSTR * lista;
	int cuenta = 0;

	// Inicializa la librería COM
	hr = CoInitialize(NULL);
	if (FAILED(hr)) {
		printf("No se pudo inicializar la librería COM\n");
		return NULL;
	}

	// Crea el Filter Graph Manager
	hr = CoCreateInstance(CLSID_CaptureGraphBuilder2, NULL,
						CLSCTX_INPROC_SERVER, IID_ICaptureGraphBuilder2,
						(void **)&pCaptureGraph);
	if (FAILED(hr)) {
		printf("No se pudo crear el Filter Graph Manager");
		CoUninitialize();
		return NULL;
	}

	// Crea el Device Enumerator
	hr = CoCreateInstance(CLSID_SystemDeviceEnum, NULL, 
						CLSCTX_INPROC_SERVER, IID_ICreateDevEnum,
						(void **)&pSysDevEnum);
	if (FAILED(hr)) {
		printf("No se pudo crear el Device Enumerator");		
		CoUninitialize();
		return NULL;
	}

	// Crea el enumerador de dispositivos de captura de video
	hr = pSysDevEnum->CreateClassEnumerator(CLSID_VideoInputDeviceCategory,
										&pEnumCat, 0);

	if (hr == S_OK) {
		IMoniker *pMoniker = NULL;
		ULONG cFetched;		
		
		// Cuenta el número de dispositivos conectados	
		cuenta = 0;		
		while(pEnumCat->Next(1, &pMoniker, &cFetched) == S_OK) {
			cuenta++;			
		}
	
		if (cuenta <= 0) {			
			pEnumCat->Release();
			pSysDevEnum->Release();
			CoUninitialize();
			return NULL;
		}
		pEnumCat->Reset();				

		// Crea la lista de dispositivos
		lista = new LPWSTR[cuenta];	

		// Recorre los dispositivos
		int it = -1;
		while (pEnumCat->Next(1, &pMoniker, &cFetched) == S_OK) {			
			IPropertyBag *pPropBag;

			it++;
			hr = pMoniker->BindToStorage(0, 0, IID_IPropertyBag,
										(void**)&pPropBag);
			if (SUCCEEDED(hr)) {
				// Obtenemos el Friendly Name del Filtro
				VARIANT varName;
				VariantInit(&varName);
				hr = pPropBag->Read(L"FriendlyName", &varName, 0);
				
				if( SUCCEEDED(hr) ){
					/*int tamano = WideCharToMultiByte(CP_ACP, 0, varName.bstrVal,
													SysStringLen(varName.bstrVal), 
													NULL, 0, NULL, NULL);
					char *cadena = (char *)alloca( tamano + 1 ); 
					cadena[0] = 0;
					WideCharToMultiByte( CP_ACP, 0, varName.bstrVal,
										SysStringLen(varName.bstrVal), 
										cadena, tamano, NULL, NULL );
					SysFreeString(varName.bstrVal);																				
					cadena[tamano] = '\0';*/
					
					int tamano = wcslen(varName.bstrVal) + 3;
					lista[it] = new wchar_t[tamano];
					wsprintf(lista[it], L"%s:%d", varName.bstrVal, it);					
				}			
				VariantClear(&varName);				
				pPropBag->Release();			
			}			
			pMoniker->Release();			
		}				
		pEnumCat->Release();		
	}

	//Destructores	
	pSysDevEnum->Release();
	CoUninitialize();

	*tamanoLista = cuenta;
		
	return lista;

}

/***************************************************************
// Función que inicializa la apertura de la memoria compartida
***************************************************************/
int CCapturaVLC::initMC(LPWSTR dispositivo) {
	
	map<LPWSTR, manejador_t>::iterator iter = imgDispositivos.find(dispositivo);	
	if (iter != imgDispositivos.end()) {
		fprintf( stderr, "Error, el dispositivo %s ya ha obtenido la memoria compartida", dispositivo);
		return FUNCION_FALLA;
	}

	// Calcula el tamaño de la memoria compartida en la primera vuelta
	int tamano = sizeof(int) + sizeof(int) + sizeof(char) * 4;
	manejador_t manejador;
	manejador.formato = new char[5];

	// Abre la zona de memoria compartida
	HANDLE hMapFile = OpenFileMapping(
                   FILE_MAP_ALL_ACCESS,   // read/write access
                   FALSE,                 // do not inherit the name
                   dispositivo);               // name of mapping object 
 
	if (hMapFile == NULL) {
		fprintf(stderr, "No se pudo abrir el objeto de mapeado %S, %d\n", dispositivo, GetLastError());
		return FUNCION_FALLA;
	} 
   unsigned char * pBuf = (unsigned char *) MapViewOfFile(hMapFile, // handle to map object
               FILE_MAP_READ,  
               0,                    
               0,                    
               tamano);                   
 
   if (pBuf == NULL) { 
   		fprintf(stderr, "No se pudo abrir el fichero de mapeado, %d\n", GetLastError());
		return FUNCION_FALLA;
   }

	int it = 0;		
	CopyMemory(&(manejador.ancho), (int *)&(pBuf[it]), sizeof(int));
	it += sizeof(int);
	CopyMemory(&(manejador.alto), (int *)&(pBuf[it]), sizeof(int));
	it += sizeof(int);		
	CopyMemory(manejador.formato, (char *)&(pBuf[it]), sizeof(char) * 4);
	it += sizeof(char) * 4;
	manejador.formato[4] = '\0';	
	manejador.devicename = dispositivo;
	UnmapViewOfFile(pBuf);

	int imagesize = manejador.ancho * manejador.alto * 3;
	tamano += sizeof(unsigned char) * imagesize;
	int posicion = it;
		
	// Abre definitivamente la memoria compartida
	pBuf = (unsigned char *) MapViewOfFile(hMapFile, // handle to map object
               FILE_MAP_READ,  
               0,                    
               0,                    
               tamano);
	if (pBuf == NULL) { 
		fprintf(stderr, "No se pudo abrir el fichero de mapeado final, %d\n", GetLastError());
		return FUNCION_FALLA;
	}
	manejador.hMapFile = hMapFile;
	manejador.imagesize = imagesize;
	manejador.pBuf = pBuf;
	manejador.posicion = posicion;

	imgDispositivos[dispositivo] = manejador;

	return EXITO;
}

/***************************************************************
// Función que libera la memoria compartida para un dispositivo
***************************************************************/
int CCapturaVLC::endMC(LPWSTR dispositivo) {
	map<LPWSTR, manejador_t>::iterator iter = imgDispositivos.find(dispositivo);	
	if (iter == imgDispositivos.end()) {
		fprintf( stderr, "Error, el dispositivo %s no ha obtenido la memoria compartida", dispositivo);
		return FUNCION_FALLA;
	}
	HANDLE hMapFile;
	unsigned char * pBuf;	
	try {
		manejador_t manejador = iter->second;
		hMapFile = manejador.hMapFile;
		pBuf = manejador.pBuf;
		delete manejador.formato;
		UnmapViewOfFile((void *)pBuf);
		CloseHandle(hMapFile);
		imgDispositivos.erase(iter);
	} catch (char * str) {
		fprintf (stderr, "Excepción al liberar memoria compartida, %s\n", str);
	}		

	return EXITO;
}

/***********************************************************************
// Función que libera la memoria compartida para todos los dispositivos
************************************************************************/
int CCapturaVLC::endMC() {
	HANDLE hMapFile;
	unsigned char * pBuf;
	for( map<LPWSTR, manejador_t>::iterator iter = imgDispositivos.begin(); 
			iter != imgDispositivos.end(); iter++ ) {
		try {
			manejador_t manejador = iter->second;
			hMapFile = manejador.hMapFile;
			pBuf = manejador.pBuf;
			delete manejador.formato;
			UnmapViewOfFile((void *)pBuf);
			CloseHandle(hMapFile);
		} catch (char * str) {
			fprintf (stderr, "Excepción al liberar memoria compartida, %s\n", str);
		}
	}

	imgDispositivos.clear();

	return EXITO;
}

/***********************************************************************
// Función que captura la imagen desde la zona de memoria compartida
************************************************************************/
IplImage * CCapturaVLC::captura(LPWSTR dispositivo) {
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
		
		CopyMemory(imagen, (unsigned char *)&(pBuf[posicion]), sizeof(unsigned char) * imagesize);

		IplImage * ipl = cvCreateImage(cvSize(ancho, alto), IPL_DEPTH_8U, 3);
		ipl->origin = 1;

		cvZero(ipl);
		for (int i = 0; i < alto; i++) {
			for (int j = 0; j < ancho; j++) {				
				CV_IMAGE_ELEM(ipl, unsigned char, i, j * 3) = (unsigned int)imagen[i * ancho * 3 + j * 3];
				CV_IMAGE_ELEM(ipl, unsigned char, i, j * 3 + 1) = (unsigned int)imagen[i * ancho * 3 + j * 3 + 1];
				CV_IMAGE_ELEM(ipl, unsigned char, i, j * 3 + 2) = (unsigned int)imagen[i * ancho * 3 + j * 3 + 2];
			}
		}

		delete imagen;

		return ipl;
	} catch (char * str) {
		fprintf (stderr, "Excepción al obtener la imagen, %s\n", str);
	}

	return NULL;
}