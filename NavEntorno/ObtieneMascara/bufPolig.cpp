#include "StdAfx.h"
#include "bufPolig.h"

bufPolig::bufPolig(void) {
	maxBuffer = 5;
}

bufPolig::~bufPolig(void) {
	buffer.clear();	
}

tPoligono bufPolig::getPoligono(tPoligono nuevo) {	
	buffer.push_back(nuevo);
	acotar();
	
	tPoligono media = { cvPoint(0, 0), cvPoint(0, 0), cvPoint(0, 0), cvPoint(0, 0), cvPoint(0, 0) };
	int div = 0;
	for (int i = 0; i < buffer.size(); i++) {
		tPoligono it = (tPoligono)buffer.at(i);
		media.p1.x += it.p1.x * (i + 1);
		media.p1.y += it.p1.y * (i + 1);
		media.p2.x += it.p2.x * (i + 1);
		media.p2.y += it.p2.y * (i + 1);
		media.p3.x += it.p3.x * (i + 1);
		media.p3.y += it.p3.y * (i + 1);
		media.p4.x += it.p4.x * (i + 1);
		media.p4.y += it.p4.y * (i + 1);
		media.p5.x += it.p5.x * (i + 1);
		media.p5.y += it.p5.y * (i + 1);

		div += i + 1;
	}

	media.p1.x /= div;
	media.p1.y /= div;
	media.p2.x /= div;
	media.p2.y /= div;
	media.p3.x /= div;
	media.p3.y /= div;
	media.p4.x /= div;
	media.p4.y /= div;
	media.p5.x /= div;
	media.p5.y /= div;

	return media;
}

void bufPolig::setMaxBuffer(int val){
	maxBuffer = val;
	acotar();
}

void bufPolig::acotar() {	
	while(buffer.size() > maxBuffer) {		    
		buffer.erase(buffer.begin());
	}
}
