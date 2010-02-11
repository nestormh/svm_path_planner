/*
 * Ssc.cpp
 *
 *  Created on: 21/12/2009
 *      Author: jonatan
 */

#include "Ssc.h"

Ssc::Ssc(int serial) {
	char port[15];

	sprintf (port, "/dev/ttyS%d", serial);

	printf ("%s\n", port);

	fd=open(port, O_RDWR | O_NOCTTY | O_NDELAY);

	if (fd == -1 ){

		perror("open_port: Unable to open /dev/ttyS0 – ");

	} else {

		fcntl(fd, F_SETFL,0);

		printf("Port %d has been sucessfully opened and %d is the file description\n",serial, fd);

	}
}

void Ssc::move(int servo, int pos) {
	char command[10];
	int wr;

	sprintf (command, "%c%c%c", 255, servo, pos);

	wr=write(fd,command,3);

}

Ssc::~Ssc() {
	// TODO Auto-generated destructor stub
	close(fd);
}
