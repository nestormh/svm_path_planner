/*
 * Creado el 18/02/2008
 *
 *Creado por Alberto Hamilcon con Eclipse
 */
package sibtra.lms;

/**
 * @author alberto
 *
 * Objeto sin métodos para contener un barrido en coordenadas cartesianas
 */
public class BarridoCartesiano {
	
	/**
	 * vector con las componentes x. 
	 * Estas podrán ser positivas o negativas
	 */
	public short	x[];
	
	/**
	 * vector con las componentes y.
	 * Estas sólo deberán ser positivas.
	 */
	public short	y[];
	
	/**
	 * Si es verdadero la distancia es en milímetros, si es falso en centímetros
	 */
	public boolean	enMilimetros;

}
