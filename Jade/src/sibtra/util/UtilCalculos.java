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
	 * se devuelve el valor umbralPositivo, si está decreciendo se devuelve 0.
	 * Si está dentro de la zona muerta negativa y está decreciendo se devuelve el umbralNegativo, si está 
	 * creciendo se devuelve 0.
	 * @param valor valor actual de la variable
	 * @param anterior valor anterior de la variable
	 * @param umbralPositivo umbral positivo de la zona muerta
	 * @param umbralNegativo umbral negativo de la zona muerta
	 * @return el valor umbralizado
	 */
	public static double zonaMuertaCon0(double valor, double anterior, double umbralPositivo, double umbralNegativo) {
		//comprobaciones básicas
		if(umbralPositivo<=0)
			throw new IllegalArgumentException("Valor de zona muerta positiva ("+umbralPositivo+") ha de se positivo");
		if(umbralNegativo>=0)
			throw new IllegalArgumentException("Valor de zona muerta negativa ("+umbralNegativo+") ha de se negativo");
		//aplicamos la lógica
		if(valor>umbralPositivo)
			return valor;
		if(valor<umbralNegativo)
			return valor;
		
		if(valor==0.0)
			return 0;
		
		if(valor>0) //estamos en la zona muerta positiva
			if(valor>anterior) //estamos creciendo
				return umbralPositivo;
			else
				return 0;
		else //estamos en zona muerta negativa
			if(valor<anterior) //estamos decreciendo
				return umbralNegativo;
			else
				return 0;
	}
	
	/**
	 * Implementa una zona muerta con 0.
	 * Si el valor está fuera de la zona muerta se devulve sin modificar. 
	 * Si el valor está dentro de la zona muerta, es positivo y está creciendo 
	 * se devuelve el valor umbralPositivo, si está decreciendo se devuelve 0.
	 * Si está dentro de la zona muerta negativa y está decreciendo se devuelve el umbralNegativo, si está 
	 * creciendo se devuelve 0.
	 * @param valor valor actual de la variable
	 * @param anterior valor anterior de la variable
	 * @param umbralPositivo umbral positivo de la zona muerta
	 * @param umbralNegativo umbral negativo de la zona muerta
	 * @return el valor umbralizado
	 */
	public static int zonaMuertaCon0(int valor, int anterior, int umbralPositivo, int umbralNegativo) {
		if(valor>umbralPositivo)
			return valor;
		if(valor<umbralNegativo)
			return valor;
		
		if(valor==0)
			return 0;
		
		if(valor>0) //estamos en la zona muerta positiva
			if(valor>anterior) //estamos creciendo
				return umbralPositivo;
			else
				return 0;
		else //estamos en zona muerta negativa
			if(valor<anterior) //estamos decreciendo
				return umbralNegativo;
			else
				return 0;
	}

	/** zonaMuertaCon0 con el mismo umbral para la parte positiva y negativa */
	public static double zonaMuertaCon0(double valor, double anterior, double umbral) {
		return zonaMuertaCon0(valor, anterior, umbral, -umbral);
	}

	/** zonaMuertaCon0 con el mismo umbral para la parte positiva y negativa */
	public static int zonaMuertaCon0(int valor, int anterior, int umbral) {
		return zonaMuertaCon0(valor, anterior, umbral, -umbral);
	}

	/** @return devuelve entero SIN SIGNO tomando a como byte para parte alta y b como parte baja */
	public static int byte2entero(int a, int b) {
		return a * 256 + b;
	}

	/** @return devuelve entero CON SIGNO tomando a como byte para parte alta y b como parte baja */
	public static int byte2enteroSigno(int a, int b) {
		if (a > 129) {
			int atemp = a * 256 + b - 1;
			atemp = atemp ^ 65535;
	
			return -1 * atemp;
		}
	
		return a * 256 + b;
	}

	/**
	 * @param v1 primer verctor de 2 componentes (x,y)
	 * @param v2 segundo vector de 2 componenetes (x,y)
	 * @return angulo formado por los 2 vectores en rango (-PI,PI)
	 */
	public static double anguloVectores(double[] v1, double[] v2) {
		double Ti1=Math.atan2(v1[1],v1[0]);
		double Ti2=Math.atan2(v2[1],v2[0]);
		double Ti=Ti2-Ti1;
		return normalizaAngulo(Ti);
	}

	/** @return distancia ecuclídea entre p1 y p2	 */
	public static double distanciaPuntos(double[] p1, double[] p2) {
		double[] d={p1[0]-p2[0], p1[1]-p2[1]};
		return largoVector(d);
	}

	/** @return largo euclídeo del vector */
	public static double largoVector(double[] d) {
		return Math.sqrt(d[0]*d[0]+d[1]*d[1]);
	}
	
	/** @return el minimo de entre min y los valores en vect */
	public static double minimo(double min, double[] vect) {
		if(vect!=null && vect.length>0)
			for(int i=0; i<vect.length; i++)
				if(vect[i]<min)
					min=vect[i];
		return min;
	}
	
	/** @return el maximo de entre max y los valores en vect */
	public static double maximo(double max, double[] vect) {
		if(vect!=null && vect.length>0)
			for(int i=0; i<vect.length; i++)
				if(vect[i]>max)
					max=vect[i];
		return max;
	}

	/**
	 * Devuelve mínimo de entre min y el mínimo de columna ind del vector v.
	 * @param ind columna a comprobar
	 * @param min minimo inicial
	 * @param v vector cuya columna se va a recorrer
	 * @return mínimo de entre min y el mínimo de columna ind del vector v
	 */
	protected static double min(int ind,double min, double[][] v) {
		if(v!=null && v.length>0 && v[0].length>=ind)
			for(int i=0; i<v.length; i++)
				if(v[i][ind]<min)
					min=v[i][ind];
		return min;
	}

	/**
	 * @return Mínimo de la columna ind de los 3 vectores pasados
	 */
	protected static double min(int ind, double[][] v1, double[][] v2, double[][] v3) {
		return min(ind,min(ind,min(ind,java.lang.Double.POSITIVE_INFINITY,v1),v2),v3);
	}

	/**
	 * Devuelve maximo de entre max y el máximo de columna ind del vector v.
	 * @param ind columna a comprobar
	 * @param max máximo inicial
	 * @param v vector cuya columna se va a recorrer
	 * @return maximo de entre max y el máximo de columna ind del vector v
	 */
	protected static double max(int ind,double max, double[][] v) {
		if(v!=null && v.length>0 && v[0].length>=ind)
			for(int i=0; i<v.length; i++)
				if(v[i][ind]>max)
					max=v[i][ind];
		return max;
	}

	/** @return Máximo de la columna ind de los 3 vectores pasados */
	protected static double max(int ind, double[][] v1, double[][] v2, double[][] v3) {
		return max(ind,max(ind,max(ind,java.lang.Double.NEGATIVE_INFINITY,v1),v2),v3);
	}
}
