#pragma once

class bufPolig
{
private:
	vector<tPoligono> buffer;
	int maxBuffer;
	void acotar();
public:	
	bufPolig(void);
	~bufPolig(void);
	tPoligono getPoligono(tPoligono nuevo);
	void setMaxBuffer(int val);
};
