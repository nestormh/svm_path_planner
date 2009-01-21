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
	//printf ("Fin de función \n");
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
//	printf ("Sale del destructor\n");*/
}

/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN: Inserta una línea en la estructura. Hace las comprobaciones adecuadas para:
	            - Evitar duplicados
	            - Evitar líneas contenidas en otras
	            - Evitar paralelas muy cercanas
	PARÁMETROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
void Lineas::Insert(CvPoint *item, int pos) {
	int j;
	CvPoint *aux;
	bool insert, 
		 direction;
	
	insert = true;

	if (abs(item[0].x - item[1].x) < abs(item[0].y - item[1].y)){	// Comprobar la dirección de la línea
		direction = true;		// Vertical
	}else{
		direction = false;		// Horizontal
	}	
	
	if (lines[pos]->first == NULL) {	// Si no hay ninguna línea en esa posición	
		index[n] = pos;
		n++;	
	} else {
		j = 0;
		do {							// Comprobar si ya está insertado
			aux = (CvPoint *) cvGetSeqElem(lines[pos], j);
			if ((item[0].x == aux[0].x) && (item[0].y == aux[0].y) 
				&& (item[1].x == aux[1].x) && (item[1].y == aux[1].y)) { 
				insert = false;				// Si está no se repite
				printf ("No se inserta (repetida)\n");				
			} else if (((item[0].x >= aux[0].x) && (item[1].x <= aux[1].x))		// La nueva está contenida en una existente 
				|| ((item[0].y >= aux[0].y) && (item[1].y <= aux[1].y))) {
				insert = false;
				printf ("No se inserta (contenida)\n");
			} else if (((item[0].x <= aux[0].x) && (item[1].x >= aux[1].x)) 	// La nueva contiene a una existente -> se deja la mayor
				|| ((item[0].y <= aux[0].y) && (item[1].y >= aux[1].y))) {
			printf("Modificada \n");
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
void Lineas::Insert(CvPoint *item, int pos, int ventana) {
	int i;
//	CvPoint *aux;
	bool direction,
	     insert;
	int lado;
//	int lx, ly;
	
	if (abs(item[0].x - item[1].x) < abs(item[0].y - item[1].y)){	// Comprobar la dirección de la línea
		direction = true;		// Vertical
	}else{
		direction = false;		// Horizontal
	}	
	
	printf ("Insertando [(%d, %d) (%d, %d)]-> ", item[0].x, item[0].y, item[1].x, item[1].y);
	
	insert = true;
	lado = round(ventana / 2);
	i = pos + lado;
	
	// Mirar si hay algo en el entorno escogido
	while ((i >= pos - lado) && (i > 0) && (lines[i]->first == NULL)){
		i--;
	}
	
	if ((i < 0) || (i <= pos - lado) || ((i >= 0) && (lines[i]->first == NULL))){	// Si no hay ninguna línea en ese entorno	
		index[n] = pos;
		n++;	
		cvSeqPush(lines[pos], item);
		printf ("Sin modificar\n");
	} else {
		if (direction) {
			item[0].x = i;
			item[1].x = i;
		} else {
			item[0].y = i;
			item[1].y = i;
		}
		printf ("[(%d, %d) (%d, %d)]\n", item[0].x, item[0].y, item[1].x, item[1].y);
		this->Insert(item, i);
		
//		j = 0;
//		do {							// Comprobar si ya está insertado
//			aux = (CvPoint *) cvGetSeqElem(lines[i], j);
//			lx = MAX (abs(item[0].x - item[1].x), abs(aux[0].x - aux[1].x));
//			ly = MAX (abs(item[0].y - item[1].y), abs(aux[0].y - aux[1].y));
//			if (((abs(item[0].x - aux[0].x) <= lado) && (abs(item[0].y - aux[0].y) <= round(ly * 0.8))) 
//				|| ((abs(item[1].x - aux[1].x) <= round(0.8 * lx)) && (abs(item[1].y - aux[1].y) <= lado))) 
//				insert = false;				// Si está no se repite
//
//			j++;
//		}while (j < lines[i]->total);
	
//		if (insert){
//			index[n] = pos;
//			n++;	
//			cvSeqPush(lines[pos], item);
//		}
	}
}

/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN: El objetivo de esta función es insertar las líneas de manera que sólo haya una 
	            línea dentro del entorno de la ventana. Para ello, si al ir a insertar se encuentra
	            otra línea dentro de la ventana se combinan ambas, manteniendo la posición de la
	            insertada en primer lugar.          
	PARÁMETROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
void Lineas::InsertGreedy(CvPoint *item, int pos, int ventana) {
	int i;
	CvPoint *aux;
	bool direction,
		 insert;
	int lado;
		
	if (abs(item[0].x - item[1].x) < abs(item[0].y - item[1].y)){	// Comprobar la dirección de la línea
		direction = true;		// Vertical
	}else{
		direction = false;		// Horizontal
	}	

	insert = true;
	lado = round(ventana / 2);
	i = pos + lado;
	
//	printf ("Insertando (%d, %d) (%d, %d)-> ", item[0].x, item[0].y, item[1].x, item[1].y);
	
	while ((i >= pos - lado) && (i > 0) && (lines[i]->first == NULL)){		// Comprobar entorno, deja i en el índice adecuado (posición a insertar o )
		i--;
	}
	
	if ((i < 0) || (i <= pos - lado) || ((i >= 0) && (lines[i]->first == NULL))){	// Si no hay ninguna línea en ese entorno	
		index[n] = pos;
		n++;	
		cvSeqPush(lines[pos], item);
//		printf ("Queda Igual\n");
	} else {
		aux = (CvPoint *) cvGetSeqElem(lines[i], 0); // No se saca porque se asumen ordenadas de mayor a menor
		if (direction) {		// Si vertical
			aux[0].y = MAX (item[0].y, aux[0].y);
			aux[1].y = MIN (item[1].y, aux[1].y);
		} else {	// Si horizontal
			aux[0].x = MAX (item[0].x, aux[0].x);
			aux[1].x = MIN (item[1].x, aux[1].x);
		}
		
//		printf ("Queda (%d, %d) (%d, %d)\n", aux[0].x, aux[0].y, aux[1].x, aux[1].y);
		
		item[0].x = aux[0].x;
		item[0].y = aux[0].y;
		item[1].x = aux[1].x;
		item[1].y = aux[1].y;
	}
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

/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
void Lineas::DrawLines(IplImage* imagen, CvScalar color) {
	int i, j;
	CvPoint *aux;

	for (i = 0; i < n; i ++) {
		j = 0;
		do {
			aux = (CvPoint *) cvGetSeqElem(lines[index[i]], j);
			cvLine( imagen, aux[0], aux[1], color, 1, 8 );
			
			j++;
		}while (j < lines[index[i]]->total);
	}
}
