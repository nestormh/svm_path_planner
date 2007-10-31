#ifndef _INTERFAZ_LIB_VLC_21_NOV_2006
#define _INTERFAZ_LIB_VLC_21_NOV_2006

#include <windows.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <map>

using namespace std;
// Definición de las constantes de VLC

#define VLC_SUCCESS			0			// Sin error
#define VLC_ENOMEM			-1			// Memoria insuficiente
#define VLC_ETHREAD			-2			// Error de Thread
#define VLC_ETIMEOUT		-3			// Fuera de tiempo
#define VLC_ENOMOD			-10			// Módulo no localizado
#define VLC_ENOOBJ			-20			// Objeto no encontrado
#define VLC_EBADOBJ			-21			// Tipo de objeto inadecuado
#define VLC_ENOVAR			-30			// Variable no encontrada
#define VLC_EBADVAR			-31			// Valor de variable erróneo
#define VLC_EEXIT			-255		// El programa finalizó
#define VLC_EEXITSUCCESS	-999		// Se salió del programa con éxito
#define VLC_EGENERIC		-666		// Error genérico
#define VLC_FALSE			0	
#define VLC_TRUE			1
#define PLAYLIST_INSERT		1
#define PLAYLIST_APPEND		2
#define PLAYLIST_GO			4
#define PLAYLIST_PREPARSE	8
#define PLAYLIST_END		-666

#define EXITO				0
#define NO_LIB				-1
#define NO_ENCONTRADO		-2
#define FUNCION_FALLA		-3

// Definición de tipos propios de VLC
typedef int vlc_bool_t;
typedef struct vlc_list_t vlc_list_t;
typedef struct vlc_object_t vlc_object_t;
typedef signed __int64 vlc_int64_t;

typedef union{
	int             i_int;
	vlc_bool_t      b_bool;
	float           f_float;
	char *          psz_string;
	void *          p_address;
	vlc_object_t *	p_object;
	vlc_list_t *    p_list;
	vlc_int64_t     i_time;
	 
	struct { 
		char *psz_name; 
		int i_object_id; 
	} var;

	/* Make sure the structure is at least 64bits */
	struct { char a, b, c, d, e, f, g, h; } padding;
	
} vlc_value_t;

// Definición de los tipos de las funciones
/************************************************************************

char const * VLC_Version( void );
char const * VLC_Error( int i_err );
int VLC_Create( void );
int VLC_Init( int i_object, int i_argc, char *ppsz_argv[] );
int VLC_AddIntf( int i_object, char const *psz_module, 
					vlc_bool_t b_block, vlc_bool_t b_play );
int VLC_Die( int i_object );
int VLC_CleanUp( int i_object );
int VLC_Destroy( int i_object )
int VLC_VariableSet( int i_object, char const *psz_var, vlc_value_t value );
int VLC_VariableGet( int i_object, char const *psz_var, vlc_value_t *p_value );
int VLC_VariableType( int i_object, char const *psz_var, int *pi_type );
int VLC_AddTarget( int i_object, char const *psz_target,
                   char const **ppsz_options, int i_options,
                   int i_mode, int i_pos );
int VLC_Play( int i_object );
int VLC_Pause( int i_object );
int VLC_Stop( int i_object );
vlc_bool_t VLC_IsPlaying( int i_object );
float VLC_PositionGet( int i_object );
float VLC_PositionSet( int i_object, float i_position );
int VLC_TimeGet( int i_object );
int VLC_TimeSet( int i_object, int i_seconds, vlc_bool_t b_relative );
int VLC_LengthGet( int i_object );
float VLC_SpeedFaster( int i_object );
float VLC_SpeedSlower( int i_object );
int VLC_PlaylistIndex( int i_object );
int VLC_PlaylistNumberOfItems( int i_object );
int VLC_PlaylistNext( int i_object );
int VLC_PlaylistPrev( int i_object );
int VLC_PlaylistClear( int i_object );
int VLC_VolumeSet( int i_object, int i_volume );
int VLC_VolumeGet( int i_object );
int VLC_VolumeMute( int i_object );
int VLC_FullScreen( int i_object );

*************************************************************************/

typedef const char * (*_VLC_VERSION)(void);
typedef const char * (*_VLC_ERROR)(int);
typedef int (*_VLC_CREATE)(void);
typedef int (*_VLC_INIT)(int, int, char **);
typedef int (*_VLC_ADDINTF)(int, const char *, vlc_bool_t, vlc_bool_t);
typedef int (*_VLC_DIE)(int);
typedef int (*_VLC_CLEANUP)(int);
typedef int (*_VLC_DESTROY)(int);
typedef int (*_VLC_VARIABLESET)(int, char const *, vlc_value_t);
typedef int (*_VLC_VARIABLEGET)(int, char const *, vlc_value_t *);
typedef int (*_VLC_VARIABLETYPE)(int, char const *, int *);
typedef int (*_VLC_ADDTARGET)(int, char const *, char const **, int, int, int);
typedef int (*_VLC_PLAY)(int);
typedef int (*_VLC_PAUSE)(int);
typedef int (*_VLC_STOP)(int);
typedef vlc_bool_t (*_VLC_ISPLAYING)(int);
typedef float (*_VLC_POSITIONGET)(int);
typedef float (*_VLC_POSITIONSET)(int, float);
typedef int (*_VLC_TIMEGET)(int);
typedef int (*_VLC_TIMESET)(int, int, vlc_bool_t);
typedef int (*_VLC_LENGTHGET)(int);
typedef float (*_VLC_SPEEDFASTER)(int);
typedef float (*_VLC_SPEEDSLOWER)(int);
typedef int (*_VLC_PLAYLISTINDEX)(int);
typedef int (*_VLC_PLAYLISTNUMBEROFITEMS)(int);
typedef int (*_VLC_PLAYLISTNEXT)(int);
typedef int (*_VLC_PLAYLISTPREV)(int);
typedef int (*_VLC_PLAYLISTCLEAR)(int);
typedef int (*_VLC_VOLUMESET)(int, int);
typedef int (*_VLC_VOLUMEGET)(int);
typedef int (*_VLC_VOLUMEMUTE)(int);
typedef int (*_VLC_FULLSCREEN)(int);

// Definición de variables globales para acceder a las funciones
_VLC_VERSION VLC_Version;
_VLC_ERROR VLC_Error;
_VLC_CREATE VLC_Create;
_VLC_INIT VLC_Init;
_VLC_ADDINTF VLC_AddIntf;
_VLC_DIE VLC_Die;
_VLC_CLEANUP VLC_CleanUp;
_VLC_DESTROY VLC_Destroy;
_VLC_VARIABLESET VLC_VariableSet;
_VLC_VARIABLEGET VLC_VariableGet;
_VLC_VARIABLETYPE VLC_VariableType;
_VLC_ADDTARGET VLC_AddTarget;
_VLC_PLAY VLC_Play;
_VLC_PAUSE VLC_Pause;
_VLC_STOP VLC_Stop;
_VLC_ISPLAYING VLC_IsPlaying;
_VLC_POSITIONGET VLC_PositionGet;
_VLC_POSITIONSET VLC_PositionSet;
_VLC_TIMEGET VLC_TimeGet;
_VLC_TIMESET VLC_TimeSet;
_VLC_LENGTHGET VLC_LengthGet;
_VLC_SPEEDFASTER VLC_SpeedFaster;
_VLC_SPEEDSLOWER VLC_SpeedSlower;
_VLC_PLAYLISTINDEX VLC_PlaylistIndex;
_VLC_PLAYLISTNUMBEROFITEMS VLC_PlaylistNumberOfItems;
_VLC_PLAYLISTNEXT VLC_PlaylistNext;
_VLC_PLAYLISTPREV VLC_PlaylistPrev;
_VLC_PLAYLISTCLEAR VLC_PlaylistClear;
_VLC_VOLUMESET VLC_VolumeSet;
_VLC_VOLUMEGET VLC_VolumeGet;
_VLC_VOLUMEMUTE VLC_VolumeMute;
_VLC_FULLSCREEN VLC_FullScreen;

HINSTANCE libvlc;



//HANDLE hMapFile;
//unsigned char * pBuf;

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
map <LPWSTR, manejador_t> imgDispositivos;


#endif
