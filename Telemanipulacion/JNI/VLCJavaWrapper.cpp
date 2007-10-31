// VLCMedia.cpp: define las rutinas de inicialización del archivo DLL.
//

#include "stdafx.h"
#include "InterfazLibVLC.h"
#include "carrito_media_Media.h"
#include "carrito_media_MediaCanvas.h"
#include <string.h>
#include "dshow.h"
#include <stdio.h>
#include "malloc.h"
#include <jni.h>
#include <jawt_md.h>
#include <string.h>

/**********************************************************************
 * loadLibrary: Carga la librería VLC
 **********************************************************************/
int loadLibrary() {
	if (libvlc == NULL) {
		libvlc = LoadLibrary(L"libvlc");	
	}
	if (libvlc != NULL) {
		return EXITO;
	} else {
		printf("Libreria libvlc.dll no encontrada\n");
		return NO_LIB;
	}
}

/**********************************************************************
 * freeLibrary: Libera la librería VLC
 **********************************************************************/
bool freeLibrary() {
	BOOL libera = false;
	if (libvlc != NULL) {
		printf("Liberando libreria libvlc.dll\n");
		libera = FreeLibrary(libvlc);
		if (!libera) {
			printf("Error al liberar\n");
		}
	}
	return (libera == VLC_TRUE);
}

/**********************************************************************
 * funcionFalla: Muestra un mensaje si una función no puede ser 
 * obtenida desde la DLL
 **********************************************************************/
inline int funcionFalla(const char * msg) {
	printf("La funcion %s no pudo inicializarse\n", msg);
	freeLibrary();
	return FUNCION_FALLA;
}

/**********************************************************************
 * initLib: Carga todas las funciones desde la librería
 **********************************************************************/
int initLib() {

	if (libvlc == NULL) {
		if (loadLibrary() != EXITO) {
			return NO_LIB;
		}			
	}

	if ((VLC_AddIntf = (_VLC_ADDINTF) GetProcAddress(libvlc, "VLC_AddIntf")) == NULL) 
		return funcionFalla("VLC_AddIntf");
	if ((VLC_AddTarget = (_VLC_ADDTARGET) GetProcAddress(libvlc, "VLC_AddTarget")) == NULL) 
		return funcionFalla("VLC_AddTarget");
	if ((VLC_CleanUp = (_VLC_CLEANUP) GetProcAddress(libvlc, "VLC_CleanUp")) == NULL)
		return funcionFalla("VLC_CleanUp");
	if ((VLC_Create = (_VLC_CREATE) GetProcAddress(libvlc, "VLC_Create")) == NULL)
		return funcionFalla("VLC_Create");
	if ((VLC_Destroy = (_VLC_DESTROY) GetProcAddress(libvlc, "VLC_Destroy")) == NULL)
		return funcionFalla("VLC_Destroy");
	if ((VLC_Die = (_VLC_DIE) GetProcAddress(libvlc, "VLC_Die")) == NULL)
		return funcionFalla("VLC_Die");
	if ((VLC_Error = (_VLC_ERROR) GetProcAddress(libvlc, "VLC_Error")) == NULL)
		return funcionFalla("VLC_Error");
	if ((VLC_FullScreen = (_VLC_FULLSCREEN) GetProcAddress(libvlc, "VLC_FullScreen")) == NULL)
		return funcionFalla("VLC_FullScreen");
	if ((VLC_Init = (_VLC_INIT) GetProcAddress(libvlc, "VLC_Init")) == NULL)
		return funcionFalla("VLC_Init");
	if ((VLC_IsPlaying = (_VLC_ISPLAYING) GetProcAddress(libvlc, "VLC_IsPlaying")) == NULL)
		return funcionFalla("VLC_IsPlaying");
	if ((VLC_LengthGet = (_VLC_LENGTHGET) GetProcAddress(libvlc, "VLC_LengthGet")) == NULL) 
		return funcionFalla("VLC_LengthGet");
	if ((VLC_Pause = (_VLC_PAUSE) GetProcAddress(libvlc, "VLC_Pause")) == NULL)
		return funcionFalla("VLC_Pause");
	if ((VLC_Play = (_VLC_PLAY) GetProcAddress(libvlc, "VLC_Play")) == NULL)
		return funcionFalla("VLC_Play");
	if ((VLC_PlaylistClear = (_VLC_PLAYLISTCLEAR) GetProcAddress(libvlc, "VLC_PlaylistClear")) == NULL)
		return funcionFalla("VLC_PlaylistClear");
	if ((VLC_PlaylistIndex = (_VLC_PLAYLISTINDEX) GetProcAddress(libvlc, "VLC_PlaylistIndex")) == NULL)
		return funcionFalla("VLC_PlaylistIndex");
	if ((VLC_PlaylistNext = (_VLC_PLAYLISTNEXT) GetProcAddress(libvlc, "VLC_PlaylistNext")) == NULL)
		return funcionFalla("VLC_PlaylistNext");
	if ((VLC_PlaylistNumberOfItems = (_VLC_PLAYLISTNUMBEROFITEMS) GetProcAddress(libvlc, "VLC_PlaylistNumberOfItems")) == NULL)
		return funcionFalla("VLC_PlaylistNumberOfItems");
	if ((VLC_PlaylistPrev = (_VLC_PLAYLISTPREV) GetProcAddress(libvlc, "VLC_PlaylistPrev")) == NULL)
		return funcionFalla("VLC_PlaylistPrev");
	if ((VLC_PositionGet = (_VLC_POSITIONGET) GetProcAddress(libvlc, "VLC_PositionGet")) == NULL)
		return funcionFalla("VLC_PositionGet");
	if ((VLC_PositionSet = (_VLC_POSITIONSET) GetProcAddress(libvlc, "VLC_PositionSet")) == NULL)
		return funcionFalla("VLC_PositionSet");
	if ((VLC_SpeedFaster = (_VLC_SPEEDFASTER) GetProcAddress(libvlc, "VLC_SpeedFaster")) == NULL) 
		return funcionFalla("VLC_SpeedFaster");
	if ((VLC_SpeedSlower = (_VLC_SPEEDSLOWER) GetProcAddress(libvlc, "VLC_SpeedSlower")) == NULL)
		return funcionFalla("VLC_SpeedSlower");
	if ((VLC_Stop = (_VLC_STOP) GetProcAddress(libvlc, "VLC_Stop")) == NULL)
		return funcionFalla("VLC_Stop");
	if ((VLC_TimeGet = (_VLC_TIMEGET) GetProcAddress(libvlc, "VLC_TimeGet")) == NULL)
		return funcionFalla("VLC_TimeGet");
	if ((VLC_TimeSet = (_VLC_TIMESET) GetProcAddress(libvlc, "VLC_TimeSet")) == NULL) 
		return funcionFalla("VLC_TimeSet");
	if ((VLC_VariableGet = (_VLC_VARIABLEGET) GetProcAddress(libvlc, "VLC_VariableGet")) == NULL)
		return funcionFalla("VLC_VariableGet");
	if ((VLC_VariableSet = (_VLC_VARIABLESET) GetProcAddress(libvlc, "VLC_VariableSet")) == NULL) 
		return funcionFalla("VLC_VariableSet");
	if ((VLC_VariableType = (_VLC_VARIABLETYPE) GetProcAddress(libvlc, "VLC_VariableType")) == NULL)
		return funcionFalla("VLC_VariableType");
	if ((VLC_Version = (_VLC_VERSION) GetProcAddress(libvlc, "VLC_Version")) == NULL)
		return funcionFalla("VLC_Version");
	if ((VLC_VolumeGet = (_VLC_VOLUMEGET) GetProcAddress(libvlc, "VLC_VolumeGet")) == NULL)
		return funcionFalla("VLC_VolumeGet");
	if ((VLC_VolumeMute = (_VLC_VOLUMEMUTE) GetProcAddress(libvlc, "VLC_VolumeMute")) == NULL)
		return funcionFalla("VLC_VolumeMute");
	if ((VLC_VolumeSet = (_VLC_VOLUMESET) GetProcAddress(libvlc, "VLC_VolumeSet")) == NULL)
		return funcionFalla("VLC_VolumeSet");
	return EXITO;
}

/**********************************************************************
 * verError: Muestra un mensaje de error
 **********************************************************************/
void verError(int error) {
	fprintf(stderr, "Error: %s\n", (VLC_Error)(error));
}

/**********************************************************************
 * Funciones propias de la librería JNI:
 **********************************************************************/

/**********************************************************************
 * cargaLibreria: Carga la librería VLC dentro de la clase
 **********************************************************************/
JNIEXPORT jboolean JNICALL 
Java_carrito_media_Media__1cargaLibreria(JNIEnv *env , jobject obj) {
	if (initLib() != EXITO) {
		printf("No se pudo cargar la libreria\n");
		return false;
	}
	return true;
}

/**********************************************************************
 * creaInstancia: Crea una nueva instancia multimedia
 **********************************************************************/
JNIEXPORT jint JNICALL 
Java_carrito_media_Media__1creaInstancia(JNIEnv * env, jobject, jobjectArray arr) {
	char ** comandos;
	jsize len = env->GetArrayLength(arr);
	comandos = new char*[len];
	
	for (int i = 0; i < len; i++) {
		jobject obj = env->GetObjectArrayElement(arr, i);
		jclass str = env->GetObjectClass(obj);
		jmethodID length = env->GetMethodID(str, "length", "()I");
		jmethodID at = env->GetMethodID(str, "charAt", "(I)C");
	
		int size = env->CallIntMethod(obj, length);
		comandos[i] = new char[size + 1];
		comandos[i][size] = '\0';
		for (int j = 0; j < size; j++) {
			comandos[i][j] = char(env->CallCharMethod(obj, at, j));
		}		
	}
	
	// Ahora que ya tenemos el char **, llamamos a la función en la libreria VLC
	int id = -1;

	printf("Creando instancia de VLC\n");
    int i_ret = (VLC_Create)();
    if( i_ret < 0 ) {
		verError(i_ret);
        return i_ret;
    }	

	id = i_ret;
    /* Inicializa libvlc */
    i_ret = (VLC_Init)( id, len, comandos );
    if( i_ret < 0 ) {
		verError(i_ret);
        (VLC_Destroy)( id );
        return i_ret == VLC_EEXITSUCCESS ? 0 : i_ret;
    }	
	return id;
}

/**********************************************************************
 * play: Inicia la reproducción
 **********************************************************************/
JNIEXPORT jboolean JNICALL 
Java_carrito_media_Media__1play(JNIEnv * env, jobject, jint id) {	
	int i_ret = (VLC_Play)(id);
	if( i_ret < 0 ) {
		verError(i_ret);  
		return false;
	}
	return true;
}

/**********************************************************************
 * pausa: Pausa la reproducción
 **********************************************************************/
JNIEXPORT jboolean JNICALL 
Java_carrito_media_Media__1pausa(JNIEnv *, jobject, jint id) {
	int i_ret = (VLC_Pause)(id);
	if( i_ret < 0 ) {
		verError(i_ret); 
		return false;
	}
	return true;
}

/**********************************************************************
 * stop: Detiene la reproducción
 **********************************************************************/
JNIEXPORT jboolean JNICALL 
Java_carrito_media_Media__1stop(JNIEnv *, jobject, jint id) {
	int i_ret = (VLC_Stop)(id);
	if( i_ret < 0 ) {
		verError(i_ret); 
		return false;
	}
	return true;
}

/**********************************************************************
 * fullScreen: Reproduce el stream a pantalla completa
 **********************************************************************/
JNIEXPORT jboolean JNICALL 
Java_carrito_media_Media__1fullScreen(JNIEnv *, jobject, jint id) {
	int i_ret = (VLC_FullScreen)(id);
	if( i_ret < 0 ) {
		verError(i_ret); 
		return false;
	}
	return true;
}

/**********************************************************************
 * isPlaying: Comprueba si se está reproduciendo
 **********************************************************************/
JNIEXPORT jboolean JNICALL 
Java_carrito_media_Media__1isPlaying(JNIEnv *, jobject, jint id) {
	if ((VLC_IsPlaying)(id))
		return true;
	return false;
}

/**********************************************************************
 * getLength: Comprueba la longitud del stream
 **********************************************************************/
JNIEXPORT jint JNICALL 
Java_carrito_media_Media__1getLength(JNIEnv *, jobject, jint id) {
	int i_ret = (VLC_LengthGet)(id);
	if (i_ret < 0) {
		verError(i_ret);
	}
	return i_ret;
}

/**********************************************************************
 * clearPlayList: Vacía la lista de reproducción
 **********************************************************************/
JNIEXPORT jboolean JNICALL 
Java_carrito_media_Media__1clearPlaylist(JNIEnv *, jobject, jint id) {
	int i_ret = (VLC_PlaylistClear)(id);
	if( i_ret < 0 ) {
		verError(i_ret); 
		return false;
	}
	return true;
}

/**********************************************************************
 * getPlaylistIndex: Obtiene el índice actual dentro de la lista de 
 * reproducción
 **********************************************************************/
JNIEXPORT jint JNICALL 
Java_carrito_media_Media__1getPlaylistIndex(JNIEnv *, jobject, jint id) {
	int i_ret = (VLC_PlaylistIndex)(id);
	if( i_ret < 0 ) {
		verError(i_ret); 		
	}
	return i_ret;
}

/**********************************************************************
 * nextPlaylist: Avanza dentro de la lista de reproducción
 **********************************************************************/
JNIEXPORT jboolean JNICALL 
Java_carrito_media_Media__1nextPlaylist(JNIEnv *, jobject, jint id) {
	int i_ret = (VLC_PlaylistNext)(id);
	if( i_ret < 0 ) {
		verError(i_ret); 
		return false;
	}
	return true;
}

/**********************************************************************
 * lastPlaylist: Retrocede dentro de la lista de reproducción
 **********************************************************************/
JNIEXPORT jboolean JNICALL 
Java_carrito_media_Media__1lastPlaylist(JNIEnv *, jobject, jint id) {
	int i_ret = (VLC_PlaylistPrev)(id);
	if( i_ret < 0 ) {
		verError(i_ret); 
		return false;
	}
	return true;
}

/**********************************************************************
 * getPlayListLength: Obtiene la longitud de la lista de reproducción
 **********************************************************************/
JNIEXPORT jint JNICALL 
Java_carrito_media_Media__1getPlayListLength (JNIEnv *, jobject, jint id) {
	int i_ret = (VLC_PlaylistNumberOfItems)(id);
	if( i_ret < 0 ) {
		verError(i_ret); 		
	}
	return i_ret;
}

/**********************************************************************
 * getPos: Obtiene la posición dentro del stream
 **********************************************************************/
JNIEXPORT jfloat JNICALL 
Java_carrito_media_Media__1getPos(JNIEnv *, jobject, jint id) {
	float f_ret = (VLC_PositionGet)(id);
	if( f_ret < 0 ) {
		verError(int(f_ret)); 		
	}
	return f_ret;
}

/**********************************************************************
 * setPos: Establece la posición dentro del stream
 **********************************************************************/
JNIEXPORT jboolean JNICALL 
Java_carrito_media_Media__1setPos(JNIEnv *, jobject, jint id, jfloat pos) {
	float f_ret = (VLC_PositionSet)(id, pos);
	if( f_ret < 0 ) {
		verError(int(f_ret)); 
		return false;
	}
	return true;
}

/**********************************************************************
 * setFaster: Hace que la reproducción sea más rápida
 **********************************************************************/
JNIEXPORT jboolean JNICALL 
Java_carrito_media_Media__1setFaster(JNIEnv *, jobject, jint id) {
	float f_ret = (VLC_SpeedFaster)(id);
	if( f_ret < 0 ) {
		verError(int(f_ret)); 
		return false;
	}
	return true;
}

/**********************************************************************
 * setSlower: Hace que la reproducción sea más lenta
 **********************************************************************/
JNIEXPORT jboolean JNICALL 
Java_carrito_media_Media__1setSlower(JNIEnv *, jobject, jint id){
	float f_ret = (VLC_SpeedSlower)(id);
	if( f_ret < 0 ) {
		verError(int(f_ret)); 
		return false;
	}
	return true;
}

/**********************************************************************
 * getTime: Obtiene la posición en segundos dentro del stream
 **********************************************************************/
JNIEXPORT jint JNICALL 
Java_carrito_media_Media__1getTime(JNIEnv *, jobject, jint id) {
	int i_ret = (VLC_TimeGet)(id);
	if( i_ret < 0 ) {
		verError(i_ret); 		
	}
	return i_ret;
}

/**********************************************************************
 * setTime: Fija la posición en segundos dentro del stream
 **********************************************************************/
JNIEXPORT jboolean JNICALL 
Java_carrito_media_Media__1setTime(JNIEnv *, jobject, jint id, jint seconds, jboolean relative){
	int i_ret = (VLC_TimeSet)(id, seconds, relative);
	if( i_ret < 0 ) {
		verError(i_ret); 
		return false;
	}
	return true;
}

/**********************************************************************
 * getVolume: Obtiene el volumen de sonido
 **********************************************************************/
JNIEXPORT jint JNICALL 
Java_carrito_media_Media__1getVolume(JNIEnv *, jobject, jint id) {
	int i_ret = (VLC_VolumeGet)(id);
	if( i_ret < 0 ) {
		verError(i_ret); 		
	}
	return i_ret;
}

/**********************************************************************
 * setVolume: Establece el volumen de sonido
 **********************************************************************/
JNIEXPORT jboolean JNICALL 
Java_carrito_media_Media__1setVolume(JNIEnv *, jobject, jint id, jint volume) {
	int i_ret = (VLC_VolumeSet)(id, volume);
	if( i_ret < 0 ) {
		verError(i_ret); 		
	}
	return i_ret;
}

/**********************************************************************
 * setMute: Silencia el sonido
 **********************************************************************/
JNIEXPORT jboolean JNICALL 
Java_carrito_media_Media__1setMute(JNIEnv *, jobject, jint id) {
	int i_ret = (VLC_VolumeMute)(id);
	if( i_ret < 0 ) {
		verError(i_ret); 		
	}
	return i_ret;
}



/**********************************************************************
 * eliminaInstancia: Elimina una instancia multimedia
 **********************************************************************/
JNIEXPORT jboolean JNICALL 
Java_carrito_media_Media__1eliminaInstancia(JNIEnv *, jobject, jint id) {
	// Finaliza los threads
    int i_ret = (VLC_CleanUp)( id );
	if( i_ret < 0 ) {
		verError(i_ret); 
		return false;
	}
    // Destruye la estructura libvlc
    i_ret = (VLC_Destroy)( id );
	if( i_ret < 0 ) {
		verError(i_ret); 
		return false;
	}

	return true;
}

/**********************************************************************
 * liberaLibreria: Libera la librería VLC
 **********************************************************************/
JNIEXPORT jboolean JNICALL 
Java_carrito_media_Media__1liberaLibreria(JNIEnv *, jobject) {
	return freeLibrary();
}

/**********************************************************************
 * listaDispositivos: Obtiene una lista de los dispositivos de 
 * captura conectados
 **********************************************************************/
JNIEXPORT jobjectArray JNICALL 
Java_carrito_media_Media__1listaDispositivos(JNIEnv * env, jclass cls){
#ifdef WIN32                        // Obtiene la lista para Windows
	// Variables necesarias
	HRESULT hr = 0;
	ICaptureGraphBuilder2 *pCaptureGraph = NULL;	
	IBaseFilter *pVideoInputFilter = NULL;
	ICreateDevEnum *pSysDevEnum = NULL;
	IEnumMoniker *pEnumCat = NULL;	
	char ** lista;
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
		lista = new char*[cuenta];	

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
					int tamano = WideCharToMultiByte(CP_ACP, 0, varName.bstrVal,
													SysStringLen(varName.bstrVal), 
													NULL, 0, NULL, NULL);
					char *cadena = (char *)alloca( tamano + 1 ); 
					cadena[0] = 0;
					WideCharToMultiByte( CP_ACP, 0, varName.bstrVal,
										SysStringLen(varName.bstrVal), 
										cadena, tamano, NULL, NULL );
					SysFreeString(varName.bstrVal);																				
					cadena[tamano] = '\0';
										
					lista[it] = cadena;						
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
		
	// Lo pasa al formato comprensible por Java
	jobjectArray arr = env->NewObjectArray(cuenta,  
						env->FindClass("java/lang/String"),
						env->NewStringUTF(""));
	for (int i = 0; i < cuenta; i++) {		
		env->SetObjectArrayElement(arr, i, 
			env->NewStringUTF(lista[i]));
	}	

	return arr;
#else				// Para otras versiones, escribir aquí
	// TODO Si se desea una versión para Linux, escribir aquí
#endif
}

/**********************************************************************
 * paint: Dibuja el vídeo en un Canvas de Java
 **********************************************************************/
JNIEXPORT void JNICALL 
Java_carrito_media_MediaCanvas_paint (JNIEnv * env, jobject canvas, jobject graphics, jint id) {
  JAWT awt;
  JAWT_DrawingSurface* ds;
  JAWT_DrawingSurfaceInfo* dsi;
  JAWT_Win32DrawingSurfaceInfo* dsi_win;

  jint lock;
  bool isLocked = false;
    
  vlc_value_t value;

  try {
	// Obtiene AWT
	awt.version = JAWT_VERSION_1_3;
	if (JAWT_GetAWT(env, &awt) == JNI_FALSE) {
	    printf("AWT no encontrado\n");
		return;
	}

	// Obtiene la superficie de dibujo
	ds = awt.GetDrawingSurface(env, canvas);
	if (ds == NULL) {
	    printf("Superficie de dibujo nula\n");
		return;
	}

	// Bloquea la superficie de dibujo
	lock = ds->Lock(ds);
	isLocked = true;
	
	if((lock & JAWT_LOCK_ERROR) != 0) {
		printf("Error al bloquear la superficie de dibujo\n");
		awt.FreeDrawingSurface(ds);
		return;
	}

	// Obtiene la información de la superficie de dibujo
	dsi = ds->GetDrawingSurfaceInfo(ds);
	if (dsi == NULL) {
	    printf("Error al obtener la información de la superficie de dibujo\n");
	    ds->Unlock(ds);
		awt.FreeDrawingSurface(ds);
		return;
	} 

	// Obtiene la información de la superficie de dibujo específica de la plataforma
#ifdef WIN32	// Versión para Windows
	dsi_win = (JAWT_Win32DrawingSurfaceInfo*)dsi->platformInfo;
	value.i_int = reinterpret_cast<int>(dsi_win->hwnd); 
#else			// Para Linux sería algo así. NOTA: No ha sido probado
	dsi_win = (JAWT_X11DrawingSurfaceInfo*)dsi->platformInfo;
	value.i_int = reinterpret_cast<int>(dsi_win->drawable); 
#endif

	// Dibuja
	VLC_VariableSet( id, "drawable", value );  

	// Libera la información de la superficie de dibujo
	ds->FreeDrawingSurfaceInfo(dsi);

	// Desbloquea la superficie de dibujo
	ds->Unlock(ds);
	isLocked = false;

	/* Libera la superficie de dibujo */
	awt.FreeDrawingSurface(ds);	
  } catch (char * str) {
	  fprintf(stderr, "Error al dibujar la imagen, %s", str);
	  if (ds != NULL) {
		if (dsi != NULL) {
			ds->FreeDrawingSurfaceInfo(dsi);
		}
		if (isLocked) {
			ds->Unlock(ds);
		}
		awt.FreeDrawingSurface(ds);
	  }
  }
}

/**********************************************************************
 * initMC: Abre la memoria compartida
 **********************************************************************/
int initMC(LPWSTR dispositivo) {

	map<LPWSTR, manejador_t>::iterator iter = imgDispositivos.find(dispositivo);	
	if (iter != imgDispositivos.end()) {
		fprintf( stderr, "Error, el dispositivo %s ya ha obtenido la memoria compartida", dispositivo);
		return FUNCION_FALLA;
	}
	int tamano = sizeof(int) + sizeof(int) + sizeof(char) * 4;
	manejador_t manejador;
	manejador.formato = new char[5];

	HANDLE hMapFile = OpenFileMapping(
                   FILE_MAP_ALL_ACCESS,   // read/write access
                   FALSE,                 // do not inherit the name
                   dispositivo);               // name of mapping object 
 
	if (hMapFile == NULL) {
		fprintf(stderr, "No se pudo abrir el objeto de mapeado, %d\n", GetLastError());
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

/**********************************************************************
 * endMC: Cierra la memoria compartida
 **********************************************************************/
JNIEXPORT jint JNICALL 
Java_carrito_media_Media__1endMC(JNIEnv *, jobject) {
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
	return EXITO;
}

/**********************************************************************
 * getAncho: Obtiene el ancho de la imagen
 **********************************************************************/
JNIEXPORT jint JNICALL 
Java_carrito_media_Media__1getAncho (JNIEnv * env, jobject, jstring midisp) {
	try {
		const char* cadena = env->GetStringUTFChars(midisp, NULL);
		int len = env->GetStringUTFLength(midisp);
		LPWSTR dispositivo = new wchar_t[len + 1];
		for (int i = 0; i < len; i++) {
			dispositivo[i] = cadena[i];
		}
		dispositivo[len] = L'\0';

		map<LPWSTR, manejador_t>::iterator iter = imgDispositivos.find(dispositivo);	
		if (iter == imgDispositivos.end()) {
			int ret = initMC(dispositivo);
			if (ret != EXITO) {
				fprintf(stderr, "Error al obtener el ancho\n");
				return FUNCION_FALLA;
			}
		}	

		iter = imgDispositivos.find(dispositivo);	
		manejador_t manejador;
		if (iter == imgDispositivos.end()) {
			fprintf(stderr, "El iterador no se actualizo\n");
			return FUNCION_FALLA;
		} else {
			manejador = (manejador_t)iter->second;
		}

		return manejador.ancho;

	} catch (char * str) {
		fprintf (stderr, "Excepción al obtener el ancho, %s\n", str);
	}
	return FUNCION_FALLA;
}

/**********************************************************************
 * getAlto: Obtiene el alto de la imagen
 **********************************************************************/
JNIEXPORT jint JNICALL 
Java_carrito_media_Media__1getAlto (JNIEnv * env, jobject, jstring midisp) {
	
	try {
		const char* cadena = env->GetStringUTFChars(midisp, NULL);
		int len = env->GetStringUTFLength(midisp);
		LPWSTR dispositivo = new wchar_t[len + 1];
		for (int i = 0; i < len; i++) {
			dispositivo[i] = cadena[i];
		}
		dispositivo[len] = L'\0';

		map<LPWSTR, manejador_t>::iterator iter = imgDispositivos.find(dispositivo);	
		if (iter == imgDispositivos.end()) {
			int ret = initMC(dispositivo);
			if (ret != EXITO) {
				fprintf(stderr, "Error al obtener el alto\n");
				return FUNCION_FALLA;
			}
		}

		iter = imgDispositivos.find(dispositivo);	
		manejador_t manejador;
		if (iter == imgDispositivos.end()) {
			fprintf(stderr, "El iterador no se actualizo\n");
			return FUNCION_FALLA;
		} else {
			manejador = (manejador_t)iter->second;
		}

		return manejador.alto;

	} catch (char * str) {
		fprintf (stderr, "Excepción al obtener el alto, %s\n", str);
	}

	return FUNCION_FALLA;
}

/**********************************************************************
 * getImagen: Obtiene una imagen de un dispositivo DirectShow
 **********************************************************************/
JNIEXPORT jintArray 
JNICALL Java_carrito_media_Media__1getImagen (JNIEnv * env, jobject, jstring midisp) {

	try {
		const char* cadena = env->GetStringUTFChars(midisp, NULL);
		int len = env->GetStringUTFLength(midisp);
		LPWSTR dispositivo = new wchar_t[len + 1];
		for (int i = 0; i < len; i++) {
			dispositivo[i] = cadena[i];
		}
		dispositivo[len] = L'\0';
		delete cadena;

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

		if (formato == "RV24") {
			for (int i = 0; i < alto; i++) {
				CopyMemory(&(imagen2[i * ancho * 3]), &(imagen[(alto - i - 1) * ancho * 3]), sizeof(unsigned char) * ancho * 3);
			}

			for (int i = 0; i < imagesize; i += 3) {
				imagen[i] = imagen2[i + 2];
				imagen[i + 1] = imagen2[i + 1];
				imagen[i + 2] = imagen2[i];
			}
		}

		jintArray rgb = env->NewIntArray(imagesize);
		jint * buf = new jint[imagesize];

		for (int i = 0; i < imagesize; i++) {
			buf[i] = (jint)imagen[i];
		}

		env->SetIntArrayRegion(rgb, 0, imagesize, buf);

		delete imagen;
		delete imagen2;
		delete buf;

		return rgb;

	} catch (char * str) {
		fprintf (stderr, "Excepción al obtener la imagen, %s\n", str);
	}

	return NULL;
	
}

/*
 * Class:     carrito_media_Media
 * Method:    _saveImagenActual
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL 
Java_carrito_media_Media__1saveImagenActual  (JNIEnv * env, jobject obj, jstring str) {
}
