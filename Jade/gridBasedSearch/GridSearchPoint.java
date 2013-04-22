package gridBasedSearch;

import java.util.Vector;

import boids.Boid;
import boids.Obstaculo;

public class GridSearchPoint {
	/**
	 * indica si el punto de la rejilla pertenece al openSet (puntos susceptibles de ser explorados)
	 */
	boolean openSet = false;
	/**
	 * indica si el punto de la rejilla pertenece al closedSet (puntos ya explorados)
	 */
	boolean closedSet = false;
	/**
	 * indica si el punto está ocupado por un obstáculo
	 */
	boolean occupied = false;
	/**
	 * índice del punto que indica su posición x en la rejilla
	 */
	int xIndex;	
	/**
	 * índice del punto que indica su posición y en la rejilla
	 */
	int yIndex;
	/**
	 * posición x del punto en el mundo real (índice multiplicado por resolution)
	 */
	double xPosition;
	/**
	 * posición y del punto en el mundo real (índice multiplicado por resolution)
	 */
	double yPosition;
	/**
	 * tamaño de cada celda
	 */
	double resolution;	
	/**
	 * Celda de la rejilla desde la que el camino a llegado a este punto
	 */
	GridSearchPoint cameFrom;	

	double g_score;
	double h_score;
	double peso_marcado;
	double f_score;
	
	/** Vector donde se almacenan los boids de la bandada*/
	Vector<Boid> particulas = new Vector<Boid>();
	/** Vector con la informacion de posicion de los obstaculos del escenario*/
	Vector<Obstaculo> obstaculos = new Vector<Obstaculo>();
	/**Vector con la prediccion de la posicion de los obstaculos en un determinado instante del futuro*/
	Vector<Obstaculo> obstaculosFuturos = new Vector<Obstaculo>();
	
	public GridSearchPoint(int xIndex, int yIndex, double resolution){
		this.xIndex = xIndex;
		this.yIndex = yIndex;
		this.resolution = resolution;
		setxPosition(resolution*xIndex);
		setyPosition(resolution*yIndex);
		this.cameFrom = null;
	}
	
	public boolean isOpenSet() {
		return openSet;
	}
	public void setOpenSet(boolean openSet) {
		this.openSet = openSet;
	}
	public boolean isClosedSet() {
		return closedSet;
	}
	public void setClosedSet(boolean closedSet) {
		this.closedSet = closedSet;
	}
	public boolean isOccupied() {
		return occupied;
	}
	public void setOccupied(boolean occupied) {
		this.occupied = occupied;
	}
	public int getxIndex() {
		return xIndex;
	}
	public void setxIndex(int xIndex) {
		this.xIndex = xIndex;
	}
	public int getyIndex() {
		return yIndex;
	}
	public void setyIndex(int yIndex) {
		this.yIndex = yIndex;
	}
	public double getxPosition() {
		return xPosition;
	}
	public void setxPosition(double xPosition) {
		this.xPosition = xPosition;
	}
	public double getyPosition() {
		return yPosition;
	}
	public void setyPosition(double yPosition) {
		this.yPosition = yPosition;
	}
	
	public double getResolution() {
		return resolution;
	}

	public void setResolution(double resolution) {
		this.resolution = resolution;
	}
	
	public double getG_score() {
		return g_score;
	}

	public void setG_score(double g_score) {
		this.g_score = g_score;
	}

	public double getH_score() {
		return h_score;
	}

	public void setH_score(double h_score) {
		this.h_score = h_score;
	}
	
	public double getPeso_marcado() {
		return peso_marcado;
	}

	public void setPeso_marcado(double peso_marcado) {
		this.peso_marcado = peso_marcado;
	}

	public double getF_score() {
		return f_score;
	}

	public void setF_score(double f_score) {
		this.f_score = f_score;
	}
	
	public GridSearchPoint getCameFrom() {
		return cameFrom;
	}

	public void setCameFrom(GridSearchPoint cameFrom) {
		this.cameFrom = cameFrom;
	}
	
	public Vector<Boid> getParticulas() {
		return particulas;
	}

	public void setParticulas(Vector<Boid> particulas) {
		this.particulas = particulas;
	}

	public Vector<Obstaculo> getObstaculos() {
		return obstaculos;
	}

	public void setObstaculos(Vector<Obstaculo> obstaculos) {
		this.obstaculos = obstaculos;
	}

	public Vector<Obstaculo> getObstaculosFuturos() {
		return obstaculosFuturos;
	}

	public void setObstaculosFuturos(Vector<Obstaculo> obstaculosFuturos) {
		this.obstaculosFuturos = obstaculosFuturos;
	}
/**
 * Método que calcula la distancia euclídea entre este punto y el que se le pasa como parámetro.Son las distancias entre
 * los índices de la rejilla, no entre las posiciones reales
 * @param point Punto hasta el cual se quiere calcular la distancia euclídea
 * @return
 */
	public double distThisPoint2Point(int [] point){
		double dist = Math.sqrt((this.getxIndex() - point[0])*(this.getxIndex() - point[0]) + 
				(this.getyIndex() - point[1])*(this.getyIndex() - point[1]));
		return dist; 
	}
	
	public double distThisPoint2Point(int xIndex, int yIndex){
		double dist = Math.sqrt((this.getxIndex() - xIndex)*(this.getxIndex() - xIndex) + 
				(this.getyIndex() - yIndex)*(this.getyIndex() - yIndex));
		return dist; 
	}
	/**
	 * Método que calcula la distancia euclídea entre este punto y el que se le pasa como parámetro. entre las posiciones reales
	 * @param point Punto hasta el cual se quiere calcular la distancia euclídea
	 * @return
	 */
	public double distThisPoint2Point(double xPos,double yPos){
		double dist = Math.sqrt((this.getxPosition() - xPos)*(this.getxPosition() - xPos) + 
				(this.getyPosition() - yPos)*(this.getyPosition() - yPos));
		return dist; 
	}
}
