

#define WIDTH 320
#define HEIGHT 240
#define COLONY_SIZE 20//20
#define MAX_SECUENCE ((WIDTH == 320)? 400:600)
#define MAX_NEIGHBORS 100//100


#define SECUENCE_ADD_STATE(c,s) s->sec[++s->top].i = c.i;\
								s->sec[s->top].j = c.j

#define SECUENCE_REMOVE_STATE(c,s) c.i = s->sec[s->top].i;\
								   c.j = s->sec[s->top--].j	

#define CREATE_STATE(a,b,c) c.i = a;\
							c.j = b

#define GET_ACTUAL_STATE(col,ind,comp) comp.i = col->rastro[ind].sec[col->rastro[ind].top].i;\
									 comp.j = col->rastro[ind].sec[col->rastro[ind].top].j

#define SET_ACTUAL_STATE(col,ind,comp) col->rastro[ind].sec[++col->rastro[ind].top].i = comp.i;\
									   col->rastro[ind].sec[col->rastro[ind].top].j = comp.j
									   

#define GET_STATE_COMPONENTS(a,b,c) b = a.i;\
									c = a.j
#define SHOW_SECUENCE(s) for (int i = 0; i <= s->top; i++)\
							printf ("%d %d->", s->sec[i].i,s->sec[i].j);\
						 printf("\n")

typedef struct {
	int i;
	int j;
}componentStruct;

typedef struct {
	componentStruct sec[MAX_SECUENCE];
	int top;
}secuenceStruct;

typedef struct {
	secuenceStruct rastro[COLONY_SIZE];	
	float feromona[WIDTH][HEIGHT];
	bool terminado[COLONY_SIZE];
}colonyStruct;

float heuristic(int, int,IplImage*);
void initColony(IplImage*,IplImage*,colonyStruct*,int,int,int,int,int,int*);
void feasibleNeighbors(int,int,secuenceStruct*);
void feasibleNeighborsBacktracking(int,int,secuenceStruct*);
void randomProportional(int,colonyStruct*,componentStruct*,secuenceStruct*,IplImage*);
void pseudoRandomProportional(int,colonyStruct*,componentStruct*,secuenceStruct*,IplImage*);
void pureRandom(componentStruct*,secuenceStruct*);
void bestRoute(secuenceStruct*,IplImage*,IplImage*,colonyStruct*);
bool condicionParada(int,int);
void manageAntActivity(IplImage*,colonyStruct*,IplImage*,int,int);
void acoMetaheuristic(IplImage*,IplImage*,colonyStruct*,secuenceStruct*,int);
void startingAreas(IplImage*,IplImage*,int,int);
void setPointofAttraction(IplImage*,IplImage*,secuenceStruct*, secuenceStruct*, int*,int*,int*,int*,int*,int*,int);