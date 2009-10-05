package boids;

import java.awt.geom.AffineTransform;
import java.awt.geom.GeneralPath;
import java.io.Serializable;

import Jama.Matrix;

public class Obstaculo implements Serializable{

	/**Vector con las componentes de velocidad del boid*/
	Matrix velocidad;
	/**Vector con las componentes de posicion del boid*/
	Matrix posicion;
	/**Forma geométrica con la que se pintará el obstáculo*/
	GeneralPath triangulo;
	
	/**
	 * Constructor general
	 * @param posicion del obstáculo
	 * @param velocidad del obstáculo
	 */
	public Obstaculo(Matrix posicion, Matrix velocidad) {
		this.velocidad = velocidad;
		this.posicion = posicion;
		/**Inicialización del aspecto gráfico del cuerpo del boid*/
		float ptosX[] = {5,0,10};
		float ptosY[] = {0,5,5};
		triangulo = new GeneralPath(GeneralPath.WIND_NON_ZERO,ptosX.length);
		triangulo.moveTo (ptosX[0], ptosY[0]);

		for (int index = 1; index < ptosX.length; index++) {
		 	 triangulo.lineTo(ptosX[index], ptosY[index]);
		};
		triangulo.closePath();
		triangulo.transform(AffineTransform.getTranslateInstance(posicion.get(0,0),posicion.get(1,0)));
	}
	public Matrix getPosicion(){
		return posicion;
	}
	public void setPosicion(Matrix pos) {
		this.posicion = pos;		
	}
	public Matrix getVelocidad(){
		return posicion;
	}
	public void setVelocidad(Matrix vel) {
		this.posicion = vel;		
	}
	public GeneralPath getForma(){
		return triangulo;
	}
}
