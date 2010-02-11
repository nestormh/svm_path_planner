/*
 * Ssc.h
 *
 *  Created on: 21/12/2009
 *      Author: jonatan
 */

#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>
#include <termios.h>
#include <unistd.h>

#ifndef SSC_H_
#define SSC_H_

class Ssc {
private:
	int fd;

public:
	Ssc(int serial);
	void move(int servo, int pos);
	virtual ~Ssc();
};

#endif /* SSC_H_ */
