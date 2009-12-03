package boids;

public class TipoCamino {
	int identificador;
	int frecuencia = 0;
	double coste = 0;
	double longitud = 0;
	
	public double getLongitud() {
		return longitud;
	}
	public void setLongitud(double longitud) {
		this.longitud = longitud;
	}
	
	public void addLongitud(double longi){
		this.longitud = this.longitud + longi;
	}
	
	public double getLongitudMedia(){
		return this.longitud/this.frecuencia;
	}
	
	public TipoCamino(int id,int frec,double cost,double longi){
		this.identificador = id;
		this.frecuencia = frec;
		this.coste = cost;
		this.longitud = longi;
	}
	public double getCoste() {
		return coste;
	}
	public void setCoste(double coste) {
		this.coste = coste;
	}
	public void addCoste(double cost){
		this.coste = this.coste + cost;
	}
	public double calculaCoste(){
		return coste/frecuencia;
	}
		
	public int getIdentificador() {
		return this.identificador;
	}
	public void setIdentificador(int id) {
		this.identificador = id;
	}
	public int getFrecuencia() {
		return this.frecuencia;
	}
	public void setFrecuencia(int frec) {
		this.frecuencia = frec;
	}
	public void addFrecuencia(){
		this.frecuencia++;
	}
}
