#define LLAVE (key_t)234						/* clave de acceso */
#define SHM_SIZE sizeof(shmStruct)

typedef struct {
	int gpsOrientation;
	int leftDist;
	int rightDist;
}shmStruct;

int shmSafeGet(void);
void* shmSafeMap(int);
void shmSafeDeconnect(void*);
void shmSafeErase(int);

// operaciones con la orientación GPS

int shmReadGPSOrientation(int);
void shmWriteGPSOrientation(int,int);


// operaciones con la información visual del ACO.

void shmReadRoadInformation(int,int*,int*,int*);
void shmWriteRoadInformation(int,int,int,int);


