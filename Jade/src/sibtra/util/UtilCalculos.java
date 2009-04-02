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
	
	
	/**
	 * Implementa una zona muerta con 0.
	 * Si el valor está fuera de la zona muerta se devulve sin modificar. 
	 * Si el valor está dentro de la zona muerta, es positivo y está creciendo 
	 * se devuelve el valor humbralPositivo, si está decreciendo se devuelve 0.
	 * Si está dentro de la zona muerta negativa y está decreciendo se devuelve el humbralNegativo, si está 
	 * creciendo se devuelve 0.
	 * @param valor valor actual de la variable
	 * @param anterior valor anterior de la variable
	 * @param humbralPositivo humbral positivo de la zona muerta
	 * @param humbralNegativo humbral negativo de la zona muerta
	 * @return el valor humbralizado
	 */
	public static double zonaMuertaCon0(double valor, double anterior, double humbralPositivo, double humbralNegativo) {
		//comprobaciones básicas
		if(humbralPositivo<=0)
			throw new IllegalArgumentException("Valor de zona muerta positiva ("+humbralPositivo+") ha de se positivo");
		if(humbralNegativo>=0)
			throw new IllegalArgumentException("Valor de zona muerta negativa ("+humbralNegativo+") ha de se negativo");
		//aplicamos la lógica
		if(valor>humbralPositivo)
			return valor;
		if(valor<humbralNegativo)
			return valor;
		
		if(valor==0.0)
			return 0;
		
		if(valor>0) //estamos en la zona muerta positiva
			if(valor>anterior) //estamos creciendo
				return humbralPositivo;
			else
				return 0;
		else //estamos en zona muerta negativa
			if(valor<anterior) //estamos decreciendo
				return humbralNegativo;
			else
				return 0;
	}
	
	/**
	 * Implementa una zona muerta con 0.
	 * Si el valor está fuera de la zona muerta se devulve sin modificar. 
	 * Si el valor está dentro de la zona muerta, es positivo y está creciendo 
	 * se devuelve el valor humbralPositivo, si está decreciendo se devuelve 0.
	 * Si está dentro de la zona muerta negativa y está decreciendo se devuelve el humbralNegativo, si está 
	 * creciendo se devuelve 0.
	 * @param valor valor actual de la variable
	 * @param anterior valor anterior de la variable
	 * @param humbralPositivo humbral positivo de la zona muerta
	 * @param humbralNegativo humbral negativo de la zona muerta
	 * @return el valor humbralizado
	 */
	public static int zonaMuertaCon0(int valor, int anterior, int humbralPositivo, int humbralNegativo) {
		if(valor>humbralPositivo)
			return valor;
		if(valor<humbralNegativo)
			return valor;
		
		if(valor==0)
			return 0;
		
		if(valor>0) //estamos en la zona muerta positiva
			if(valor>anterior) //estamos creciendo
				return humbralPositivo;
			else
				return 0;
		else //estamos en zona muerta negativa
			if(valor<anterior) //estamos decreciendo
				return humbralNegativo;
			else
				return 0;
	}

	/** zonaMuertaCon0 con el mismo humbral para la parte positiva y negativa */
	public static double zonaMuertaCon0(double valor, double anterior, double humbral) {
		return zonaMuertaCon0(valor, anterior, humbral, -humbral);
	}

	/** zonaMuertaCon0 con el mismo humbral para la parte positiva y negativa */
	public static int zonaMuertaCon0(int valor, int anterior, int humbral) {
		return zonaMuertaCon0(valor, anterior, humbral, -humbral);
	}

	/** @return devuelve entero SIN SIGNO tomando a como byte para parte alta y b como parte baja */
	public static int byte2entero(int a, int b) {
		return a * 256 + b;
	}

	/** @return devuelve entero CON SIGNO tomando a como byte para parte alta y b como parte baja */
	static int byte2enteroSigno(int a, int b) {
		if (a > 129) {
			int atemp = a * 256 + b - 1;
			atemp = atemp ^ 65535;
	
			return -1 * atemp;
		}
	
		return a * 256 + b;
	}

}
