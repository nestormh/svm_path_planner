#!/bin/bash

#Script para generar ficheros de memoria compartido

#Variables iniciales
Paquete="sibtra.shm"
Clase="ShmInterface"

#Nombre de los ficheros
fVar=Variables.lst
fHdeC=shmInterface.h
fCdeC=shmInterface.c


BaseJNI="$(echo $Paquete | tr . _)_$Clase"

fHdeJNI=${BaseJNI}.h
fCdeJNI=${BaseJNI}.c

fjava=../../src/$(echo "$Paquete"|tr . /)/${Clase}.java
ind=0;

#Leemos el fichero de variables y almacenamos los datos en arrays

#cat $fVar | egrep -v "^(#|$)" | while read la

fAux=VariablesAux.lst
cat $fVar | egrep -v "^(#|$)" > $fAux


while read la

do
	tipo[$ind]=$(echo $la | cut -d' ' -f1)
	nombre[$ind]=$(echo $la | cut -d' ' -f2)
	rwC[$ind]=$(echo $la | cut -d' ' -f3)
	rwJava[$ind]=$(echo $la | cut -d' ' -f4)
	((ind+=1))
	

done < $fAux
 
rm $fAux
export numElem=$ind
export tipo
export nombre
export rwC
export rwJava

#Generamos los distintos ficheros

#Fichero java
(
echo "

package ${Paquete};

public class ${Clase} {
	
	public static native void safeGet();
	public static native void safeErase();

"
#Recorremos Variables
#declaramos get y set del tipo adecuada para cada una de ellas

ind=0
while ((ind < numElem)) 
do
	
	case ${rwJava[$ind]} in
	
	"rw" 	)
	echo "
	public static native ${tipo[$ind]} get${nombre[$ind]}();

	public static native void set${nombre[$ind]}(${tipo[$ind]} dato);
	";;
	"r" 	)
	echo "
	public static native ${tipo[$ind]} get${nombre[$ind]}();
	";;	
	"w"	)
	echo "
	public static native void set${nombre[$ind]}(${tipo[$ind]} dato);
	"
	;;
	*	) ;;	#error
	
	esac
	
	((ind+=1))
	

done
echo "
	
	static {
		System.load(System.getProperty(\"user.dir\")+\"/lib/shm/sibtra_shm_ShmInterface.so\");
	}
}
"

) > $fjava


#Fichero JNI

numElem=$ind	
(
echo "

#include <jni.h>
#include \"$fHdeC\"
#include \"$fHdeJNI\"

JNIEXPORT void JNICALL Java_${BaseJNI}_safeGet (JNIEnv *env, jobject obj) {
	shmSafeGet();
}
JNIEXPORT void JNICALL Java_${BaseJNI}_safeErase (JNIEnv *env, jobject obj) {
	shmSafeErase();
}

"
ind=0
while ((ind < numElem))
do
	case ${rwJava[$ind]} in
	
	"rw" 	)
	echo "
        JNIEXPORT j${tipo[$ind]} JNICALL	Java_${BaseJNI}_get${nombre[$ind]} (JNIEnv *env, jobject obj){
		return shmGet${nombre[$ind]}();	
	}			

        JNIEXPORT void JNICALL	Java_${BaseJNI}_set${nombre[$ind]} (JNIEnv *env, jobject obj, j${tipo[$ind]} p){
		shmSet${nombre[$ind]}(p);	
	}

	";;
	"r" 	)
	echo "        
	JNIEXPORT j${tipo[$ind]} JNICALL	Java_${BaseJNI}_get${nombre[$ind]} (JNIEnv *env, jobject obj){
		return shmGet${nombre[$ind]}();	
	}
	";;	
	"w"	)
	echo "
	JNIEXPORT void JNICALL	Java_${BaseJNI}_set${nombre[$ind]} (JNIEnv *env, jobject obj, j${tipo[$ind]} p){
		shmSet${nombre[$ind]}(p);	
	}
	"
	;;
	*	) ;;	#error
	
	esac
	
	((ind+=1))
done

)> $fCdeJNI

#Fichero .H de C

(
echo "
#ifdef __cplusplus
extern "C" {
#endif

#define LLAVE (key_t)235
#define SHM_SIZE sizeof(shmStruct)

typedef struct {
"
ind=0

while ((ind < numElem))
do
	echo "
	${tipo[$ind]} ${nombre[$ind]};"
	((ind+=1))
done

echo " 
}shmStruct; 

void shmSafeGet(void);
void shmSafeMap(void);
void shmSafeDeconnect(void);
void shmSafeErase(void);

" 

ind=0

while ((ind < numElem))
do
	case ${rwC[$ind]} in
	
	"rw" 	)
	echo "
        ${tipo[$ind]} shmGet${nombre[$ind]}(void);
        void shmSet${nombre[$ind]}(${tipo[$ind]});
	";;
	"r" 	)
	echo "        
        ${tipo[$ind]} shmGet${nombre[$ind]}(void);
	";;	
	"w"	)
	echo "
        void shmSet${nombre[$ind]}(${tipo[$ind]});
	"
	;;
	*	) ;;	#error
	
	esac


	((ind+=1))
done

echo "
#ifdef __cplusplus
}
#endif
"
)> $fHdeC

#Fichero .C de C

(
echo "
#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>

#include \"shmInterface.h\"

shmStruct*mapPointer = NULL; 
int shmid;


void shmSafeGet (void) {
	if ((shmid=shmget(LLAVE,SHM_SIZE,IPC_CREAT | 0600)) == -1) {
		printf (\"No se ha podido crear el segmento de memoria compartida\");
		exit(-1);
	}
	shmSafeMap();
}
void shmSafeMap() {
	mapPointer = (shmStruct*)shmat(shmid,0,0);
	if ((int)mapPointer == -1) {
		printf (\"Error en el mapeo de la memoria compartida.\");
		exit(-1);
	}
}
void shmSafeDeconnect() {
	if(shmdt(mapPointer) == -1) {
		printf (\"Error en la desconexion con la memoria compartida\");
		exit(-1);
	}
}
"

ind=0

while ((ind < numElem))
do
	case ${rwC[$ind]} in
	
	"rw" 	)
	echo "
        ${tipo[$ind]} shmGet${nombre[$ind]}(void) {
		if (mapPointer == NULL) {
			printf (\"Accediendo a memoria cuando el puntero no ha sido mapeado\n\");
			exit(-1);
		}
		return mapPointer->${nombre[$ind]};
	}

        void shmSet${nombre[$ind]}(${tipo[$ind]} dato) {
		if (mapPointer == NULL) {
			printf (\"Accediendo a memoria cuando el puntero no ha sido mapeado\n\");
			exit(-1);
		}
		mapPointer->${nombre[$ind]}= dato;
	}
	";;
	"r" 	)
	echo "  
      	${tipo[$ind]} shmGet${nombre[$ind]}(void) {
		if (mapPointer == NULL) {
			printf (\"Accediendo a memoria cuando el puntero no ha sido mapeado\n\");
			exit(-1);
		}
		return mapPointer->${nombre[$ind]};
	}
	";;	
	"w"	)
	echo "
	 void shmSet${nombre[$ind]}(${tipo[$ind]} dato) {
		if (mapPointer == NULL) {
			printf (\"Accediendo a memoria cuando el puntero no ha sido mapeado\n\");
			exit(-1);
		}
		mapPointer->${nombre[$ind]}= dato;
	}
	"
	;;
	*	) ;;	#error
	
	esac


	((ind+=1))
done

)> $fCdeC
