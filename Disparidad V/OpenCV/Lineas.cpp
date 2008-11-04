#include "Lineas.h"

/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
Lineas::Lineas() {
	n = 0;
	max = 0;
	storage = NULL;
	index = NULL;
	lines = NULL;
}

/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
Lineas::Lineas(int size) {
	this->n = 0;
	this->max = size;
	this->storage = cvCreateMemStorage(0);
	this->index = new int[size];
	
	this->lines = new CvSeq*[size];
	for (int i = 0; i < size; i++) {
		this->lines[i] = cvCreateSeq( CV_32SC4, sizeof(CvSeq), 2 * sizeof(CvPoint), storage);
	//	printf ("%d\n", i);
	}
	printf ("Fin de función \n");
}

/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
Lineas::~Lineas() {
//	printf ("Entra al destructor\n");
	for (int i = 0; i < max; i++)
		cvClearSeq (lines[i]);
		
	delete(lines);
	delete(index);
	cvReleaseMemStorage (&storage);
//	printf ("Sale del destructor\n");
}

/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
void Lineas::Insert(CvPoint *item, int pos) {
	int j;
	CvPoint *aux;
	bool insert;
	
	insert = true;
	
	if (lines[pos]->first == NULL) {	// Si no hay ninguna línea en esa posición	
		index[n] = pos;
		n++;	
	} else {
		j = 0;
		do {							// Comprobar si ya está insertado
			aux = (CvPoint *) cvGetSeqElem(lines[pos], j);
			if ((item[0].x == aux[0].x) && (item[0].y == aux[0].y) 
				&& (item[1].x == aux[1].x) && (item[1].y == aux[1].y)) 
				insert = false;				// Si está no se repite
								
			else if ((item[0].x >= aux[0].x) && (item[1].x <= aux[1].x)		// La nueva está contenida en una existente 
				|| (item[0].y >= aux[0].y) && (item[1].y <= aux[1].y))
				insert = false;

			else if ((item[0].x <= aux[0].x) && (item[1].x >= aux[1].x) 	// La nueva contiene a una existente -> se deja la mayor
				|| (item[0].y <= aux[0].y) && (item[1].y >= aux[1].y)) {
//			printf("-----------------Entra-----------------\n");
				aux[0] = item[0];
				aux[1] = item[1];
				insert = false;
			}

			j++;
		}while (j < lines[pos]->total);
	}
	
	if (insert)
		cvSeqPush(lines[pos], item);
	
}

/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
int Lineas::Compare (const void * a, const void * b) {
  return ( *(int*)a - *(int*)b );
}

/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
void Lineas::Sort() {
	qsort (index, n, sizeof(int), Compare);
}

/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
CvSeq *Lineas::GetLine(int pos) {
	return (lines[pos]);
}

/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
int Lineas::GetN() {
	return (n);
}

/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
int *Lineas::GetIndex() {
	return (index);
}

/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
void Lineas::Print() {
	int i, j;
	CvPoint *aux;
	
	printf ("n = %d\n (", n);
	for (i = 0; i < n; i ++) {
		printf("%d ", index[i]);
	}
	printf(")\n");
	
	for (i = 0; i < n; i ++) {
		j = 0;
		do {
			aux = (CvPoint *) cvGetSeqElem(lines[index[i]], j);
			printf ("%d, %d: [(%d, %d)(%d, %d)]\n", index[i], j, aux[0].x, aux[0].y, aux[1].x, aux[1].y);
			j++;
		}while (j < lines[index[i]]->total);
	}
}
