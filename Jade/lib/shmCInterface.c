#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>

#include "shmCInterface.h"


int shmSafeGet() {
	
int shmid;
	if((shmid=shmget(LLAVE, SHM_SIZE, IPC_CREAT | 0600)) == -1) {
		printf("No se ha podido crear el segmento de memoria compartida.");
		exit (-1);
	}
	return shmid;
} 
/********************************************/

void* shmSafeMap(int shmid){

void* p;

	p = shmat(shmid,0,0);
	if((int)p == -1) {
		printf("Error en el mapeo de la memoria compartida.");
		exit(-1);
	}
	return p;
}

/********************************************/
void shmSafeDeconnect(void* p) {

	if(shmdt(p) == -1) {
			printf(" Error en la desconexión con la memoria compartida.");
			exit(-1);
	}
}

/*********************************************/
void shmSafeErase(int shmid) {
	
	if(shmctl(shmid, IPC_RMID, 0) == -1)
		  {
			printf("Error en el borrado del segmento de memoria compartida.");
			exit(-1);
		  }
}
/***********************************************/

int shmReadGPSOrientation(int shmid) {
	
	int o;
	shmStruct *p;

	p = (shmStruct*)shmSafeMap(shmid);
	o = p->gpsOrientation;
	shmSafeDeconnect(p);
	
	return o;
}

/*************************************************/

void shmWriteGPSOrientation(int shmid, int o) {
	
	shmStruct* p;
	
	p = (shmStruct*)shmSafeMap(shmid);
	p->gpsOrientation = o;
	
	shmSafeDeconnect(p);
	
}

/*************************************************/

void shmReadRoadInformation(int shmid,int* left,int* right,int* o) {

shmStruct* p;

	p = (shmStruct*)shmSafeMap(shmid);
	*left = p->leftDist;
	*right = p->rightDist;
	*o = p->gpsOrientation;
	
	shmSafeDeconnect(p);	
}

/***************************************************/

void shmWriteRoadInformation(int shmid,int left,int right,int o) {

shmStruct* p;

	p = (shmStruct*)shmSafeMap(shmid);
	p->gpsOrientation = o;
	p->leftDist = left;
	p->rightDist = right;
	
	shmSafeDeconnect(p);
	
}