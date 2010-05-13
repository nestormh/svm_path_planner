/**************************************************************
Procedimientos del algortimo de OCH
***************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cv.h>
#include <highgui.h>
#include "aco.h"



int limiteSuperior;			// no procesamos la imagen por encima del limite superior
int limiteInferior;			// no procesamos la imagen por debajo del limite inferior
float alpha = 0.5;
float antOnlinePheromonUpdate = 0.5;
float pheromonEvaporationRatio = 0.1;
float pheromonContributionConstant = 1.0;
float heuristicContributionConstant = 1.0;
int pheromonMapWidth;
int pheromonMapHeight;
int startingAreasWidth = 10;
int iP,jP;
int initialPosX = -1, initialPosY = -1;
CvRandState randState;
float uniformProb[COLONY_SIZE];
float startingEdgeDensity = 0.01;
float globalExploration = 0.0;
int rangoExploracion = 20;
float lBest;
bool pheromonResetFlagLeft = true;
bool pheromonResetFlagRight = true;

int startingRange = 1000;
float beta = 100;		//para la contribucion segun su distancia final al pto atraccion
int finalRange = 1000;
int startingInertia = 8;
int finalInertia = 4;

int minPlottingPath = 5;

CvPoint p1;
CvPoint p2;
CvPoint p3;
CvPoint p4;

/**************************************************
Devuelve 1 o 0 en función de que exista un borde
en la imágen de entrada en las coordenadas 
indicadas.
***************************************************/

float heuristic(int i, int j,IplImage* edges){
	float aux;
	
	aux = CV_IMAGE_ELEM(edges,unsigned char,j,i);
	
		return (float)aux / 255.0;
	
}
/************************************************************************
 Busca los estados iniciales de los agentes. Los estados iniciales son los 
 píxeles centrales de los cuadrados en los que se busca.La búsqueda se hace  
 en la columna indicada por 'x' y se detiene cuando se llega a una zona con 
 la densidad de bordes suficiente o cuando se llega la fila 'refY'.
*************************************************************************/

void startingAreas(IplImage* edges,IplImage* salida,int x,int refY){

	int sum = 0;
	int w,y = 0;
	CvScalar sumAux;

	if ((x+startingAreasWidth) > pheromonMapWidth)
		w = pheromonMapWidth-x;
	else
		w = startingAreasWidth;

	float umbral = (float)(w*w*255)*startingEdgeDensity;
	
	while((sum < umbral) && (y < limiteSuperior )){
		cvSetImageROI(edges,cvRect(x,y,w,w));
		sumAux = cvSum(edges);
		sum = sumAux.val[0];
		y += 5;
	}
	if (refY == -1) refY = y;
	
	if (abs (refY-y) > startingRange) initialPosY = refY;
	else initialPosY = y+(int)floor((float)w/2.0);
	cvResetImageROI(edges);
	cvRectangle(salida,cvPoint(x,y),cvPoint(x+w,y+w),cvScalar(255),2);	
	initialPosX = x+floor((float)w/2.0);
}

/****************************************************************************
 * Inicializa la colonia 'colony', es decir para cada hormiga perteneciente a la colonia se tiene:
 * - Todos los rastros de feromona a 0
 * - Los caminos seguidos sólo contienen una posicion: la inicial de la colonia.
 * - Condición de parada a false para cada hormiga.
 * **************************************************************************/
 
void initColony(IplImage* edges, IplImage* traces,colonyStruct* colony,int w,int bordeSup,int attractionX, int attractionY,int salidaX,int* salidaY) {

	int i,j;
	componentStruct comp;
	
	srand(20);
	limiteSuperior = bordeSup;
	pheromonMapWidth = w;
	
	iP = attractionX;
	jP = attractionY;
		  
	cvSetZero(traces);
	startingAreas(edges,traces,salidaX, *salidaY);
	
	limiteInferior = initialPosY;
	*salidaY = initialPosY;
	
	pheromonMapHeight = limiteSuperior-limiteInferior;
	
	
	lBest = -1;

	for (i = 0; i < COLONY_SIZE; i++) {
		colony->rastro[i].top = -1;
		colony->terminado[i] = false;
		CREATE_STATE(initialPosX,initialPosY,comp);
		SET_ACTUAL_STATE(colony,i,comp);
	}
	
	for (i = 0; i < edges->width; i++)
		for (j = 0; j < edges->height; j++)
					colony->feromona[i][j] = 0;

}
/*************************************************************
Secuencia de posibles vecinos del agente dada su posicion, 
el punto de atraccin y el rango de exploracin.
**************************************************************/

 void feasibleNeighbors(int i,int j,secuenceStruct* s) {
	int k,ini,fin,iA,jA;
	componentStruct c;
	
	s->top = -1;

	
	jA = j+1;
	
	iA = (int)floor(iP+(float)(i-iP)/(float)(j-jP)*(jA-jP));
	//if (((float)j/(float)jP)*20.0 < 3) rangoExploracion = 3;
	//else rangoExploracion = (int)floor(((float)j/(float)jP)*20.0);
	//iA = i;
	if ((iA - rangoExploracion) < 5) 
		ini = 5; 
	else 
		ini = iA-rangoExploracion;
	if ((iA + rangoExploracion) > (WIDTH -5)) 
		fin = WIDTH-5;
	else 
		fin = iA+rangoExploracion;

	for (k = ini; k <= fin; k++) {
		
		CREATE_STATE(k,jA,c);
		SECUENCE_ADD_STATE(c,s);
	}
 }

/*********************************************************************
 * Probabilidades de los posibles vecinos si no hoy información heurística
 * ni rastros de feromona
 **********************************************************************/
 
 void pureRandom(componentStruct* next,secuenceStruct* s) {
	int k;
	
	k = rand()%(s->top+1);
	*next = s->sec[k];
 }

 /***********************************************************************
  * Probabilidades de los posibles vecinos siguiendo la regla proporcional
  * probabilista a partir de la información heurística y de los rastros de 
  * feromona presentes en ellos
  ************************************************************************/
   
 void randomProportional(int ant_id,colonyStruct* colony,componentStruct* next,secuenceStruct* s,IplImage* edges) {

	float probs[MAX_NEIGHBORS],p;
	int k,iK,jK;
	float den = 0;
	bool flag = false;
	

	
	
	// calculamos el denominador

	for (k = 0; k <= s->top; k++) {
		GET_STATE_COMPONENTS(s->sec[k],iK,jK);
		den += alpha*colony->feromona[iK][jK]+(1-alpha)*heuristic(iK,jK,edges);
		//printf ("%f\n",colony->feromona[iK][jK]);
	}
	
	// calculamos cada una de las probabilidades de los vecinos
	
	if (!den) {											// si no hay información heu. ni ferom.
		pureRandom(next,s);
	}
	else {
		//probs[] contiene las probabilidades acumuladas de los vecinos, suman 1.
		for (k = 0; k <= s->top; k++) {					//asignamos a cada vecino su probabilidad
			GET_STATE_COMPONENTS(s->sec[k],iK,jK);
			if (!k) {
				probs[k] = (alpha*colony->feromona[iK][jK]+(1-alpha)*heuristic(iK,jK,edges)) / den;
			}
			else {
				probs[k] = probs[k-1]+((alpha*colony->feromona[iK][jK]+(1-alpha)*heuristic(iK,jK,edges)) / den);
			}
		}
		
		k = 0;
		p = (float)(rand() % 100) / 100.0;				//seleccionamos uno de los vecinos en función de su probabilidad
			while(!flag && k <= (s->top)) {
				if (p <= probs[k])
					flag = true;
				else
					++k;
			}
			if (!flag) {
				printf ("Error: probabilidades incorrectas\n");
				for (int i = 0; i <= s->top;i++)
					printf ("%f|", probs[i]);
				printf ("prob: %f den: %f\n", p,den);
			}
		
		next->i = s->sec[k].i;							//se devuelve la siguiente posición
		next->j = s->sec[k].j;
		
		
	}
 }

/********************************************
 * Comprueba si se da la condicion de parada
 * en este caso se comprueba si se ha llegado
 * a un límite superior en la imagen
 ********************************************/
  
 bool condicionParada(int j) {
	if (j >= limiteSuperior)
		return true;
	else
		return false;
	
 }

/*****************************************************
 * Actualiza el rastro de feromona del agente una vez 
 * que ha alcanzado la condición de parada.
 *****************************************************/
  
 void onlineDelayedPheromonContribution(secuenceStruct* sec, IplImage* edges,colonyStruct* col) {
	 int i,sum = 0,sum2 = 0;
	 float contribution, l;

	 for (i = 0; i <= sec->top; i++) {
		sum += 255-CV_IMAGE_ELEM(edges,unsigned char,sec->sec[i].j,sec->sec[i].i);
	 }
	 sum *= beta;
	 sum2 = beta*sec->top*abs(iP - sec->sec[sec->top].i);
	 l = (float)(sum+sum2) / (float)sec->top;
	 if (lBest == -1)
		 lBest = l;
	 else if (l < lBest)
		 lBest = l;
	 if (l == lBest) contribution = pheromonContributionConstant;
	 else contribution = pheromonContributionConstant/(l-lBest);
	
	 for (i = 0; i <= sec->top; i++) {
		col->feromona[sec->sec[i].i][sec->sec[i].j] += contribution;
	}
 }
 /************************************************************
  * Evaporación de feromona. Depende del ratio de evaporación
  ************************************************************/
   
 void pheromonEvaporation(colonyStruct* colony) {
	int i,j;

	for (i = 0; i < pheromonMapWidth;i++){
		for (j = limiteInferior; j < limiteInferior+pheromonMapHeight;j++) {
			colony->feromona[i][j] = (1-pheromonEvaporationRatio)*colony->feromona[i][j];
		}
	}
 }
 
 void setPointofAttraction(IplImage* img,IplImage* img2,secuenceStruct* izq, secuenceStruct* dcha, int* attractionX,int* aRef,int* bRef,int* cRef,int* dRef,int*corte,int consigna) {
		CvPoint2D32f leftCp[500];
		CvPoint2D32f rightCp[500];
		float leftLine[4],m1,m2,b1,b2;
		float rightLine[4];
	
		//int midPoint;
		int a,b,c,d;
		int aux, divisor = 1;
		CvPoint polyPoints[4];
		CvPoint pp[4];

		//polyPoints = (CvPoint*) new(3*sizeof(CvPoint));
		if (izq->top < minPlottingPath) {
			//*aRef = -1;
			//*cRef = -1;
			//printf ("Reference lost!!!!\n");
			//cvLine(img,cvPoint(*aRef,limiteSuperior),cvPoint(*cRef,0),cvScalar(0,0,255),3);
			//*aRef;
			//*cRef;
		}
		else {
			
			for (int i = 0; i <= izq->top; i++) {
					leftCp[i].x = izq->sec[i].i;
				leftCp[i].y = izq->sec[i].j;

			}
			cvFitLine2D(leftCp,izq->top,CV_DIST_WELSCH,NULL,0,0,leftLine);
			a = floor ((limiteSuperior-leftLine[3]) / leftLine[1] * leftLine[0] + leftLine[2]);
			c = floor ((-leftLine[3]) / leftLine[1] * leftLine[0] + leftLine[2]);
			if (*aRef == -1) *aRef = a;
			if (*cRef == -1) *cRef = c;
			//if (abs(a-*aRef)<finalRange) {
				aux = (int)floor((float)abs(a-*aRef)/(float)divisor);
				if (aux == 0) aux = 1;
				else if (aux > finalInertia) aux = finalInertia;
				if (a > *aRef) *aRef += aux;
				else if (a < *aRef) *aRef -= aux;
				
			//}
			//if (abs(c-*cRef)<startingRange) {
				aux = (int)floor((float)abs(c-*cRef)/(float)divisor);
				if (aux == 0) aux = 1;
				else if (aux > startingInertia) aux = startingInertia;
				if (c > *cRef) *cRef += aux;
				else if (c < *cRef) *cRef -= aux;
			//}
				
			
		}
                
                p1 = cvPoint(*cRef,0);
                p2 = cvPoint(*aRef,limiteSuperior);
                
		cvLine(img,cvPoint(*aRef,limiteSuperior),cvPoint(*cRef,0),cvScalar(0,50,200),5);
		if (dcha->top < minPlottingPath) {
			//*bRef;
			//*dRef;
			//printf ("Reference lost!!!!\n");
			//cvLine(img,cvPoint(*bRef,limiteSuperior),cvPoint(*dRef,0),cvScalar(0,0,255),3);
		}
		else {
			
			for (int i = 0; i <= dcha->top; i++) {
			
					rightCp[i].x = dcha->sec[i].i;
				rightCp[i].y = dcha->sec[i].j;
			}
			cvFitLine2D(rightCp,dcha->top,CV_DIST_WELSCH,NULL,0,0,rightLine);
			b = floor ((limiteSuperior-rightLine[3]) / rightLine[1] * rightLine[0] + rightLine[2]);
			d = floor ((-rightLine[3]) / rightLine[1] * rightLine[0] + rightLine[2]);
			if (*bRef == -1) *bRef = b;
			if (*dRef == -1) *dRef = d;
			//if (abs(b-*bRef)<finalRange) {
				aux = (int)floor((float)abs(b-*bRef)/(float)divisor);
				if (aux == 0) aux = 1;
				else if (aux > finalInertia) aux = finalInertia;
				if (b > *bRef) *bRef += aux;
				else if (b < *bRef) *bRef -= aux;
			//}
			//if (abs(d-*dRef)<startingRange) {
				aux = (int)floor((float)abs(d-*dRef)/(float)divisor);
				
				if (aux == 0) aux = 1;
				else if (aux > startingInertia) aux = startingInertia;
				if (d > *dRef) *dRef += aux;
				else if (d < *dRef) *dRef -= aux;
			//}
			
			
		}

                p4 = cvPoint(*dRef,0);
                p3 = cvPoint(*bRef,limiteSuperior);

		cvLine(img,cvPoint(*bRef,limiteSuperior),cvPoint(*dRef,0),cvScalar(0,50,200),5);
		
		if ((izq->top >= minPlottingPath) && (dcha->top >= minPlottingPath)) {
			*attractionX = (int)floor((float)(*aRef+*bRef)/2.0);
		}
			m1 = (float)limiteSuperior/ (float)(*aRef - *cRef+0.0001);
			m2 = (float)limiteSuperior/ (float)(*bRef - *dRef+0.0001);

			b1 = -m1*(*cRef);
			b2 = -m2*(*dRef);

			aux = (int)floor((b2*m1-b1*m2)/(m1-m2));
			*corte = aux;
			if (aux > limiteSuperior) {
				polyPoints[0].x = *aRef;
				polyPoints[0].y = limiteSuperior;
				polyPoints[1].x = *bRef;
				polyPoints[1].y = limiteSuperior;
				polyPoints[2].x = *dRef;
				polyPoints[2].y = 0;
				polyPoints[3].x = *cRef;
				polyPoints[3].y = 0;
			

				
				cvFillConvexPoly(img,polyPoints,4,cvScalar(0,50,0));
				
				cvLine(img,cvPoint((int)floor((float)(*aRef+*bRef)/2.0),limiteSuperior),cvPoint((int)floor((float)(*cRef+*dRef)/2.0),0),cvScalar(0,50,200),2);
				//cvLine(img,cvPoint((int)floor((float)(*cRef+*dRef)/2.0),0),cvPoint(consigna,limiteSuperior),cvScalar(255,0,0),2);
			}
			else {
				polyPoints[0].x = (int)floor((b2-b1)/(m1-m2));
				polyPoints[0].y = aux;
				polyPoints[1].x = *dRef;
				polyPoints[1].y = 0;

				polyPoints[2].x = *cRef;
				polyPoints[2].y = 0;
				
				
				cvFillConvexPoly(img,polyPoints,3,cvScalar(0,50,0));


				cvLine(img,polyPoints[0],cvPoint((int)floor((float)(*cRef+*dRef)/2.0),0),cvScalar(0,50,200),2);
				//cvLine(img,cvPoint((int)floor((float)(*cRef+*dRef)/2.0),0),cvPoint(consigna,aux),cvScalar(255,0,0),2);
			}
		
				
				cvSet(img2,cvScalar(1));
				pp[0] = cvPoint(*aRef+40,limiteSuperior);
				pp[1] = cvPoint(*bRef-40,limiteSuperior);
				pp[2] = cvPoint(*dRef-40,0);
				pp[3] = cvPoint(*cRef+40,0);
				cvFillConvexPoly(img2,pp,4,cvScalar(0));
		
	
		
		//cvCircle(img,cvPoint(*attractionX,limiteSuperior),3,cvScalar(255,255,255),2);
		
		
		//printf("IZQ: %d %d|| %d %d ||%d %d //%d 0\n", izq->sec[0].i,izq->sec[0].j,izq->sec[izq->top].i,izq->sec[izq->top].j,a,limiteSuperior,c);
		//printf("DCHA: %d %d|| %d %d ||%d %d //%d 0\n", dcha->sec[0].i,dcha->sec[0].j,dcha->sec[dcha->top].i,dcha->sec[dcha->top].j,a,limiteSuperior,c);
		
		
		
		
		//if (abs(*attractionX - iP)>40)
		//	*attractionX = iP;
		//printf ("%d\n", *attractionX);
		

		//printf ("atraccion:%f\n", floor((float)(a+b)/2.0));		
		//free(polyPoints);
 }
 void manageAntActivity (IplImage* edges,colonyStruct* colony,IplImage* traces,int ini_ant, int final_ant) {
	
	int ant_i,/*i,j,*/finishCounter=0;
	//int k;
	int subColony; 
	componentStruct comp;
	componentStruct next;
	secuenceStruct neighbors;
	//secuenceStruct secBack;
	
	subColony = final_ant - ini_ant;
	
	while(finishCounter < subColony) {	
		for (ant_i = ini_ant; ant_i < final_ant; ant_i++){
			if (!colony->terminado[ant_i]) { 
				GET_ACTUAL_STATE(colony,ant_i,comp);
				feasibleNeighbors(comp.i,comp.j,&neighbors);
				randomProportional(ant_i,colony,&next,&neighbors,edges);
				SET_ACTUAL_STATE(colony,ant_i,next);
				//actualizar feronomona online
				//colony->feromona[next.i][next.j] += onlinePheromonContribution();
				
				CV_IMAGE_ELEM(traces,unsigned char,next.j,3*next.i+2) = CV_IMAGE_ELEM(traces,unsigned char,next.j,3*next.i+2)+50.0; 
				if (condicionParada(next.j)) {
					colony->terminado[ant_i] = true;
					++finishCounter;
					//actualizar feromona offline
					onlineDelayedPheromonContribution(&colony->rastro[ant_i],edges,colony);
				}
			}
		}
	}
 }
 void bestRoute(secuenceStruct* sec,IplImage* traces,IplImage* edges,colonyStruct* colony) {
	
	float pheromonMax;
	int maxInd,/*i,*/j;
	componentStruct next;
	secuenceStruct neighbors;
	next.i = initialPosX;
	next.j = initialPosY;
	sec->top = -1;
	while(!condicionParada(next.j)) {
	//for(i = 0; i < 20; i++) {
		SECUENCE_ADD_STATE(next,sec);
		
		cvCircle(traces,cvPoint(next.i,next.j),1,cvScalar(0,255,0),2);
		feasibleNeighbors(next.i,next.j,&neighbors);
		pheromonMax = 0.0;
		for (j = 0; j <= neighbors.top; j++) {
			if (colony->feromona[neighbors.sec[j].i][neighbors.sec[j].j] >= pheromonMax) {
				pheromonMax = colony->feromona[neighbors.sec[j].i][neighbors.sec[j].j];
				maxInd = j;
			}
		}
		next = neighbors.sec[maxInd];
	}
/*	float acum = 0;
	for (int i = 0; i < sec->top; i++) {
		acum += CV_IMAGE_ELEM(edges,unsigned char,sec->sec[i].j,sec->sec[i].i);
	}
	acum = acum / (sec->top*255);
	if (acum < 0.3)
		sec->top = -1;
*/

	
 }
 void acoMetaheuristic(IplImage* edges,IplImage* traces,colonyStruct* colony, secuenceStruct* shortestPath,int generations) {
		
		int i;
		int subColony, generationInd = 0;
		
		subColony = floor((float)(COLONY_SIZE / generations));
		for (i = 0; i < (subColony*generations);i+=subColony) {
			if (generations == 1)
				alpha = 0.5;
			else
				alpha = (float)generationInd/(float)(generations-1);
			//alpha = 0.0;
			manageAntActivity(edges,colony,traces,i,i+subColony);
			//evaporacion feromona
			pheromonEvaporation(colony);
			++generationInd;
		}
		bestRoute(shortestPath,traces, edges,colony);
 }
 

 




