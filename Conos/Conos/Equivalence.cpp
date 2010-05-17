/*
 * Equivalence.cpp
 *
 *  Created on: 19/02/2010
 *      Author: jonatan
 */

#include "Equivalence.h"

/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
Equivalence::Equivalence() {
	// TODO Auto-generated constructor stub

	this->n = 0;
	this->table = NULL;
	this->visited = NULL;
}

/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
Equivalence::Equivalence(int size) {
	// TODO Auto-generated constructor stub
	int i;

	this->n = size;

	this->table = (eq**) malloc (size *sizeof(eq*));
	this->visited = (bool*) malloc (size *sizeof(bool));

	for (i = 0; i < size; i++){
		this->table[i] = NULL;
		this->visited[i] = false;
	}


}


/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN: Se define una equivalencia unidireccional, siendo la etiqueta label1 el nodo origen y label2 el
			    destino.
	PARÁMETROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
void Equivalence::add (int label1, int label2) {
	eq *newEq, *aux, *previous;
	int i, index, equivalent;

//	newEq->equivalent = std::max(label1, label2);		// Si se quisiera hacer siempre del menor al mayor
//	index = std::min(label1, label2);

	equivalent = label1;
	index = label2;

	if (index >= this->n) {						// Aumentar el tamaño de la tabla de equivalencias
		this->table = (eq**) realloc (this->table, 2 * this->n * sizeof(eq*));
		if (!this->table) {
			printf ("Error redimensionando tabla.\n");
			exit (-1);
		}

		this->visited = (bool*) realloc (this->visited, 2 * this->n * sizeof(bool));
		if (!this->visited) {
			printf ("Error redimensionando visitados.\n");
			exit (-1);
		}


		for (i = this->n; i < 2 * this->n; i++){		// Inicializar los nuevos elementos
			this->table[i] = NULL;
			this->visited[i] = false;
		}
		this->n = this->n * 2;
	}

	newEq = (eq*) malloc (sizeof (eq));
	newEq->next = NULL;
	newEq->equivalent = equivalent;

	if (!this->table[index])
		this->table[index] = newEq;
	else {
		aux = this->table[index];
		previous = NULL;
		while ((aux->next) && (aux->equivalent < newEq->equivalent)) {				// Buscar el punto de inserción
			previous = aux;
			aux = aux->next;
		}

		if (aux->equivalent != newEq->equivalent) {		// Evitar insertar repetidos

			if (aux->equivalent > newEq->equivalent) {							// Si es en medio
				newEq->next = aux;
				if (previous)
					previous->next = newEq;
				else									// Caso de que ocupe el primer lugar
					this->table[index] = newEq;
			} else {									// Insertar el último
				aux->next = newEq;
			}
		}else {
//			printf ("Ignoro el repetido\n");
			free (newEq);
		}
	}


}

/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
void Equivalence::print(void){
	int i;
	eq *elem;

	for (i = 0; i < this->n; i++) {
		elem = this->table[i];
		printf ("[%d]->", i);
		while (elem){
			printf ("%d->", elem->equivalent);
			elem = elem->next;
		}
		printf ("//(%d)\n", this->visited[i]);
	}

}

/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
void Equivalence::resolve (void){
	int i;
	eq *list;
	eq *elem;


//	print();
	for (i = 0; i < this->n; i++){
		elem = this->table[i];
		if (this->table[i] && (!this->visited[i])){
			list = dfs (i);
			this->table[i] = list;
		}

	}
//	print();
}

/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
eq *Equivalence::dfs (int index){
	eq *list, *returned, *aux;


	this->visited[index] = true;
//	printf ("Visitado (%d)\n", index);

	list = NULL;
	if (this->table[index]) {

		while (this->table[index]) {
			aux = this->table[index];

			if (!this->visited[aux->equivalent]) {			// Si no ha sido visitado
				if (!list) {								// Insertar el primero
					list = this->table[index];
					this->table[index] = this->table[index]->next;
					list->next = NULL;
				} else {
					this->table[index] = this->table[index]->next;
					aux->next = list;
					list = aux;
				}

				returned = dfs(aux->equivalent);

				if (returned) {
					aux = returned;
					while (aux->next)
						aux = aux->next;

					aux->next = list;
					list = returned;
				}


			} else {
//				printf ("Ya estaba visitado el %d -> liberar\n", this->table[index]->equivalent);
				this->table[index] = this->table[index]->next;
				free (aux);
			}

		}

	} else
		list = NULL;

	return (list);
}


/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
int *Equivalence::buildLUT (int nLabel, int* reg){
	int *lut;
	int i;
	eq *elem;
	int _reg;


	lut = (int *) malloc ((nLabel + 1)* sizeof(int));
	_reg = 0;

//	for (i = 0; i < nLabel; i++) {
//		lut[i] = i;
//	}

	for (i = 0; i < nLabel; i++) {

		elem = this->table[i];
		if (elem) {
			lut[i] = _reg;
			while (elem){
				lut[elem->equivalent] = _reg;
				elem = elem->next;
			}
			_reg++;
		} else if (!this->visited[i]) {
			lut[i] = _reg;
			_reg++;
		}
	}
	*reg = _reg;

//	for (i = 0; i < nLabel; i++) {
//		printf("lut[%d] = %d\n", i, lut[i]);
//	}


	return(lut);
}

/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
Equivalence::~Equivalence() {
	// TODO Auto-generated destructor stub
	int i;
	eq *elem, *previous;


	for (i = 0; i < this->n; i++) {

		elem = this->table[i];
		while (elem){
			previous = elem;
			elem = elem->next;
			free (previous);
		}
	}
	free(this->table);
	free(this->visited);
}
