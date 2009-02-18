package sibtra.util;

public class UtilCalculos {

	/**
	 * Se le pasa un ángulo en radianes y devuelve ese mismo ángulo entre 
	 * -PI y PI
	 * @param angulo Ángulo a corregir
	 * @return Ángulo en radianes corregido
	 */
	public static double normalizaAngulo(double angulo){
	    angulo -= 2*Math.PI*Math.floor(angulo/(2*Math.PI));
	    if (angulo >= Math.PI)
	        angulo -= 2*Math.PI;
	    return angulo;
	}

}
