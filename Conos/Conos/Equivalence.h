/*
 * Equivalence.h
 *
 *  Created on: 19/02/2010
 *      Author: jonatan
 */

#include <stdlib.h>
#include <stdio.h>
#include <algorithm>

#ifndef EQUIVALENCE_H_
#define EQUIVALENCE_H_

class Equivalence;

typedef struct eq {
	int equivalent;
	eq *next;

} eq;


class Equivalence {

private:
	int n;
	eq **table;
	bool *visited;

public:
	Equivalence();
	Equivalence(int size);
	void add (int label1, int label2);
	void print(void);
	void resolve (void);
	eq* dfs (int);
	int* buildLUT (int nLabelm, int *reg);
	virtual ~Equivalence();
};





#endif /* EQUIVALENCE_H_ */
