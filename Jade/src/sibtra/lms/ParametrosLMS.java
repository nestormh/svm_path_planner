/**
 * 
 */
package sibtra.lms;

/**
 * Clase abstracta que contendrá la definición del rango y resolución angular y si se trabaja en 
 * milímitros. Son datos privados que se manejan con los 'seters' y 'geters'.
 * @author alberto
 *
 */
public abstract class ParametrosLMS {

	/**
	 * Que resolución angular está configurada.
	 * En centesimas de grado => posibles valores: 100, 50 ó 25.
	 */
	private short resAngular=50;
	
	/**
	 * Que rango angular usamos: 180º o 100º
	 */
	private short rangoAngular=180;
	
	/**
	 * Resolución en distancia en milimetros (true) o en cm (false)
	 */
	private boolean enMilimetros=true;
	
	/**
	 * Constructor por defecto para que sea serializable
	 */
	public ParametrosLMS(){
		
	}
	
	/**
	 * @param rangoAngular2
	 * @param resAngular2
	 * @param enMilimetros2
	 */
	public ParametrosLMS(short rangoAngular2, short resAngular2,
			boolean enMilimetros2) {
		this.rangoAngular=rangoAngular2;
		this.resAngular=resAngular2;
		this.enMilimetros=enMilimetros2;
	}

	/**
	 * Devuelve la resolución angular configurada
	 * @return the resAngular
	 */
	public short getResAngular() {
		return resAngular;
	}

	/**
	 * Devuelve la resolición angular en 1/4 de grado
	 * @return la resolición angular en 1/4 de grado
	 */
	public short getResAngularCuartos() {
		return (short)(resAngular/25);
	}
	/**
	 * Modifica la resolución angular configurada.
	 * Valores posibles: 100, 50 ó 25.
	 * @param resAngular the resAngular to set
	 */
	public void setResAngular(short resAngular) {
		//solo modificamos si son valores válidos
		if(resAngular==100 || resAngular==50 || resAngular==25)
			this.resAngular = resAngular;
	}

	/**
	 * Devuelve el rango angular configurado.
	 * Valores posibles 
	 * @return the rangoAngular
	 */
	public short getRangoAngular() {
		return rangoAngular;
	}

	/**
	 * Modifica el rango angular configurado.
	 * Valores posibles 180 o 100
	 * @param rangoAngular the rangoAngular to set
	 */
	public void setRangoAngular(short rangoAngular) {
		if(rangoAngular==180 || rangoAngular==100)
			this.rangoAngular = rangoAngular;
	}

	/**
	 * Devuelve si las distancias están en milimitros (true) o centimetros (false)
	 * @return the enMilimetros
	 */
	public boolean isEnMilimetros() {
		return enMilimetros;
	}

	/**
	 * Modifica si las distancias están en milimitros (true) o centimetros (false)
	 * @param enMilimetros the enMilimetros to set
	 */
	public void setEnMilimetros(boolean enMilimetros) {
		this.enMilimetros = enMilimetros;
	}

}
