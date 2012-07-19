package boids;

import java.awt.geom.AffineTransform;
import java.awt.geom.GeneralPath;
import java.awt.geom.Rectangle2D;
import java.io.Serializable;

import sibtra.log.LoggerArrayDoubles;
import sibtra.log.LoggerFactory;

import com.sun.org.apache.regexp.internal.recompile;

import Jama.Matrix;

public class Obstaculo implements Serializable{

	/**Vector con las componentes de velocidad del boid*/
	Matrix velocidad;
	/**Vector con las componentes de posicion del boid*/
	Matrix posicion;
	/**Vector que señala el rumbo que quiere seguir el obstáculo*/
	Matrix rumboDeseado;
	/**Forma geométrica con la que se pintará el obstáculo*/
	Rectangle2D cuadrado;
	LoggerArrayDoubles logDatosObst;
	double lado = 2;//2	

	/**
	 * Constructor general
	 * @param posicion del obstáculo
	 * @param velocidad del obstáculo
	 */
	public Obstaculo(Matrix posicion, Matrix velocidad) {
		this.velocidad = velocidad;
		this.posicion = posicion;
		/**Inicialización del aspecto gráfico del cuerpo del obstáculo*/
		cuadrado = new Rectangle2D.Double(posicion.get(0,0)-lado/2,posicion.get(1,0)-lado/2,lado,lado);
//		logDatosObst=LoggerFactory.nuevoLoggerArrayDoubles(this, "posVelObst");
//		logDatosObst.setDescripcion("Coordenadas y velocidad [x,y,vel]");
	}
	
	public Obstaculo(Matrix posicion, Matrix velocidad, Matrix rumbo) {
		this.velocidad = velocidad;
		this.posicion = posicion;
		this.rumboDeseado = rumbo;
		/**Inicialización del aspecto gráfico del cuerpo del obstáculo*/
		cuadrado = new Rectangle2D.Double(posicion.get(0,0)-lado/2,posicion.get(1,0)-lado/2,lado,lado);
//		logDatosObst=LoggerFactory.nuevoLoggerArrayDoubles(this, "posVelObst");
//		logDatosObst.setDescripcion("Coordenadas y velocidad [x,y,vel]");
	}
	/**
	 * Constructor donde se pasan las componentes de cada vector por separado
	 * @param posX Componente x del vector posición
	 * @param posY Componente y del vector posición
	 * @param velX Componente x del vector velocidad
	 * @param velY Componente y del vector velocidad
	 * @param vecRumboX Componente x del vector rumboDeseado
	 * @param vecRumboY Componente y del vector rumboDeseado
	 */
	public Obstaculo(double posX,double posY,double velX,double velY,double vecRumboX,double vecRumboY){
		double[] arrayPos = {posX,posY};
		double[] arrayVel = {velX,velY};
		double[] arrayRumbo = {vecRumboX,vecRumboY};
		this.posicion = new Matrix(arrayPos,2);
		this.velocidad = new Matrix(arrayVel,2);
		this.rumboDeseado = new Matrix(arrayRumbo,2);
		cuadrado = new Rectangle2D.Double(posicion.get(0,0)-lado/2,posicion.get(1,0)-lado/2,lado,lado);
//		logDatosObst=LoggerFactory.nuevoLoggerArrayDoubles(this, "posVelObst");
//		logDatosObst.setDescripcion("Coordenadas y velocidad [x,y,vel]");
	}
	
	public void mover(Matrix vel,double Ts){
//		this.getForma().transform(AffineTransform.getTranslateInstance(vel.get(0,0), vel.get(1,0)));
		this.setVelocidad(vel);
		this.setPosicion(this.getPosicion().plus(this.getVelocidad().times(Ts)));
//		logDatosObst.add(getPosicion().get(0,0),getPosicion().get(1,0),
//				getVelocidad().norm2());
	}
	
	public Matrix getPosicion(){
		return posicion;
	}
	public void setPosicion(Matrix pos) {
		this.posicion = pos;
		cuadrado.setRect(pos.get(0,0)-lado/2,pos.get(1,0)-lado/2,lado,lado);
	}
	public Matrix getVelocidad(){
		return velocidad;
	}
	public void setVelocidad(Matrix vel) {
		this.velocidad = vel;		
	}
	public Matrix getRumboDeseado() {
		return rumboDeseado;
	}

	public void setRumboDeseado(Matrix rumboDeseado) {
		this.rumboDeseado = rumboDeseado;
	}
	public Rectangle2D getForma(){
		return cuadrado;
	}
	
	public double getLado() {
		return lado;
	}

	public void setLado(double lado) {
		this.lado = lado;
	}
}
