/**
 * 
 */
package sibtra.imu;

import sibtra.gps.GPSData;
import sibtra.util.UtilCalculos;

/**
 * Clase para la gestión centralizada de la corrección de la declinación magnética
 * 
 * @author alberto
 *
 */
public class DeclinacionMagnetica {

	private static double declinacionLaLaguna=Math.toRadians(-(5.0+59.0/60.0));
	//Centro ULL LLA = [ 28º 28.94909', -16º 19.30155' ,   613.73]
	private static GPSData posicionLaLaguna=new GPSData(28+28.94909/60, -16-19.30155/60 ,   613.73);
	private static double declinacionIter=Math.toRadians(-(6.0+5.0/60.0));
	//Centro del ITER LLA = [ 28º  4.05182', -16º 30.72541' ,    69.46]
	private static GPSData posicionIter=new GPSData(28+4.05182/60, -16-30.72541/60 ,   73.44);

	private double declinacionAplicar=declinacionLaLaguna;

	/** Dada una posición devuelve la declinación a aplicar.
	 * Si está lejos de puntos conocidos (o es <code>null</code>) devuelve {@link Double#NaN} */
	public static double declinacionSegunPosicion(GPSData posicion) {
		if(posicion==null)
			return Double.NaN;
		posicion.calculaLocales(posicionLaLaguna);
		double[] vec={posicion.getXLocal(),posicion.getYLocal()};
		if(UtilCalculos.largoVector(vec)<5000)
			//estamos cerca de La Laguna
			 return declinacionLaLaguna;
		else {
			//probamos con el ITER
			posicion.calculaLocales(posicionIter);
			double[] vecI={posicion.getXLocal(),posicion.getYLocal()};
			if(UtilCalculos.largoVector(vecI)<5000)
				return declinacionIter;			
		}
		System.err.println("DeclinacionMagnetica: No está cerca de ningún lugar conocido");
		return Double.NaN; 
	}

	/** contructor vacío supone que estamos en La Laguna */
	public DeclinacionMagnetica() {};
	
	/** Constructor indicando la posición */
	public DeclinacionMagnetica(GPSData posicion) {
		setPosicion(posicion);
	}
	
	/** Constructor que fija la {@link #declinacionAplicar} */
	public DeclinacionMagnetica(double valor) {
		declinacionAplicar=valor;
	}
	
	/** Cambiamos la posición y con ello puede cambiar la {@link #declinacionAplicar} */
	public void setPosicion(GPSData posicion) {
		declinacionAplicar=declinacionSegunPosicion(posicion);
	}
	
	/** Establecemos la {@link #declinacionAplicar} */
	public void setDeclinacionAplicada(double valor) {
		declinacionAplicar=valor;
	}
	
	/** @return el rumbo verdadero EN RADIANES a partir de los {@link AngulosIMU}. 
	 * Si el parametro es <code>null</code> devuelve NaN */
	public double rumboVerdadero(AngulosIMU ai) {
		if(ai==null)
			return Double.NaN;
		return UtilCalculos.normalizaAngulo(Math.toRadians(ai.getYaw())+declinacionAplicar);
	}

	/** @return la {@link #declinacionAplicar}*/
	public double getDeclinacionAplicada() {
		return declinacionAplicar;
	}
	
}
