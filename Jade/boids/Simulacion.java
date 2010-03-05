package boids;

import java.util.Vector;

import Jama.Matrix;

public class Simulacion {
	/** Coordenadas a partir de las cuales se situa la bandada*/
	Matrix posInicial = new Matrix(2,1);
	/** Coordenadas del objetivo que han de alcanzar los boids*/
	Matrix objetivo = new Matrix(2,1);
	/** Máximo valor en segundos que se permite para cada simulación*/
	double tiempoMax;
	/** Tiempo invertido en que el numBoidsOk lleguen al objetivo*/
	double tiempoInvertido;
	/** Número de boids que han alcanzado el objetivo*/
	int numBoidsOk;
	/** Distancia a la que se considera que se ha alcanzado el objetivo*/
	double distOk;
	/** Número de boids que forman la bandada*/
	int tamanoBandada;
	/** Vector donde se alamacenan los boids de la bandada*/
	Vector<Boid> bandada = new Vector<Boid>();
	/** Vector donde se almacenan los boids que han alcanzado el objetivo*/
	Vector<Boid> boidsOk = new Vector<Boid>();
	/** Vector con la información de posición de los obstáculos del escenario*/
	Vector<Obstaculo> obstaculos = new Vector<Obstaculo>();
	
	//---------------------Parámetros de los boids--------------------------------
	
	double radioObstaculo = 50;
	double radioCohesion = 100;
	double radioSeparacion = 100;
	double radioAlineacion = 30;
	double pesoCohesion = 0.05;
	double pesoSeparacion = 100;
	double pesoAlineacion = 0.5;
	double pesoObjetivo = 0.1;
	double pesoObstaculo = 300;
	double pesoLider = 5;
	double velMax = 15;
	
	//-------------------------------CONSTRUCTOR---------------------------------------
	
	public Simulacion(Matrix posInicial, Matrix objetivo, double tiempoMax,
			double tiempoInvertido, int numBoidsOk, double distOk, int tamanoBandada,
			Vector<Boid> bandada, Vector<Boid> boidsOk, Vector<Obstaculo> obstaculos,
			double radioObstaculo, double radioCohesion, double radioSeparacion,
			double radioAlineacion, double pesoCohesion, double pesoSeparacion,
			double pesoAlineacion, double pesoObjetivo, double pesoObstaculo,
			double pesoLider, double velMax) {
		super();
		this.posInicial = posInicial;
		this.objetivo = objetivo;
		this.tiempoMax = tiempoMax;
		this.tiempoInvertido = tiempoInvertido;
		this.numBoidsOk = numBoidsOk;
		this.distOk = distOk;
		this.tamanoBandada = tamanoBandada;
		this.bandada = bandada;
		this.boidsOk = boidsOk;
		this.obstaculos = obstaculos;
		this.radioObstaculo = radioObstaculo;
		this.radioCohesion = radioCohesion;
		this.radioSeparacion = radioSeparacion;
		this.radioAlineacion = radioAlineacion;
		this.pesoCohesion = pesoCohesion;
		this.pesoSeparacion = pesoSeparacion;
		this.pesoAlineacion = pesoAlineacion;
		this.pesoObjetivo = pesoObjetivo;
		this.pesoObstaculo = pesoObstaculo;
		this.pesoLider = pesoLider;
		this.velMax = velMax;
	}
	
	//------------------------------GETTERS------------------------------------
	
	
	public Vector<Boid> getBandada() {
		return bandada;
	}
	public Vector<Boid> getBoidsOk() {
		return boidsOk;
	}
	public double getDistOk() {
		return distOk;
	}
	public int getNumBoidsOk() {
		return numBoidsOk;
	}
	public Matrix getObjetivo() {
		return objetivo;
	}
	public Vector<Obstaculo> getObstaculos() {
		return obstaculos;
	}
	public double getPesoAlineacion() {
		return pesoAlineacion;
	}
	public double getPesoCohesion() {
		return pesoCohesion;
	}
	public double getPesoLider() {
		return pesoLider;
	}
	public double getPesoObjetivo() {
		return pesoObjetivo;
	}
	public double getPesoObstaculo() {
		return pesoObstaculo;
	}
	public double getPesoSeparacion() {
		return pesoSeparacion;
	}
	public Matrix getPosInicial() {
		return posInicial;
	}
	public double getRadioAlineacion() {
		return radioAlineacion;
	}
	public double getRadioCohesion() {
		return radioCohesion;
	}
	public double getRadioObstaculo() {
		return radioObstaculo;
	}
	public double getRadioSeparacion() {
		return radioSeparacion;
	}
	public int getTamanoBandada() {
		return tamanoBandada;
	}
	public double getTiempoInvertido() {
		return tiempoInvertido;
	}
	public double getTiempoMax() {
		return tiempoMax;
	}
	public double getVelMax() {
		return velMax;
	}
}
