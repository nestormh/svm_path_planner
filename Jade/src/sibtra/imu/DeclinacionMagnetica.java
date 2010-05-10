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
	private static GPSData posicionLaLaguna=new GPSData(28+28.94909/60, -16-19.30155/60 ,   613.73);
	
	private static double declinacionIter=Math.toRadians(-(6.0+5.0/60.0));
	private static GPSData posicionIter=new GPSData(28+4.19024/60, -17-29.2174/60 ,   73.44);

	private double declinacionAplicar=declinacionLaLaguna;

	public DeclinacionMagnetica() {};
	
	public DeclinacionMagnetica(GPSData posicion) {
		setPosicion(posicion);
	}
	
	public DeclinacionMagnetica(double valor) {
		declinacionAplicar=valor;
	}
	
	public void setPosicion(GPSData posicion) {
		posicion.calculaLocales(posicionLaLaguna);
		double[] vec={posicion.getXLocal(),posicion.getYLocal()};
		if(UtilCalculos.largoVector(vec)<5000)
			//estamos cerca de La Laguna
			declinacionAplicar=declinacionLaLaguna;
		else {
			//probamos con el ITER
			posicion.calculaLocales(posicionIter);
			double[] vecI={posicion.getXLocal(),posicion.getYLocal()};
			if(UtilCalculos.largoVector(vecI)<5000)
				declinacionAplicar=declinacionIter;			
		}
	}
	
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

	public double getDeclinacionAplicada() {
		return declinacionAplicar;
	}
}
