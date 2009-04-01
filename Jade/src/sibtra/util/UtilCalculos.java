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

	/**
	 * Devuelve el valor acotado entre el mínimo y el máximo
	 * @param valor original 
	 * @param minimo valor permitido
	 * @param maximo valor permitido
	 * @return valor acotado
	 */
	public static int limita(int valor, int minimo, int maximo) {
		if(minimo>maximo)
			throw new IllegalArgumentException("El valor del minimo no puede ser mayor que el máximo");
		if(valor>maximo)
			return maximo;
		if(valor<minimo)
			return minimo;
		return valor;
	}
	
	/**
	 * Devuelve el valor acotado entre el mínimo y el máximo
	 * @param valor original 
	 * @param minimo valor permitido
	 * @param maximo valor permitido
	 * @return valor acotado
	 */
	public static double limita(double valor, double minimo, double maximo) {
		if(minimo>maximo)
			throw new IllegalArgumentException("El valor del minimo no puede ser mayor que el máximo");
		if(valor>maximo)
			return maximo;
		if(valor<minimo)
			return minimo;
		return valor;
	}
	
}
