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

	/**
	 * Indice del punto más cercano de ruta al punto
	 * @param ruta array con los puntos de la tractoria
	 * @param posX Coordenada x del punto
	 * @param posY Coordenada y del punto
	 * @return Índice de la ruta en el que se encuentra el punto de la 
	 * ruta más cercano al punto (posX,posY)
	 */
	public static int indiceMasCercano(double[][] ruta,double posX,double posY){
	    //Buscar punto más cercano al coche
	        double distMin=Double.POSITIVE_INFINITY;
	        int indMin=0;
	        double dx;
	        double dy;
	        for(int i=0; i<ruta.length; i++) {
	            dx=posX-ruta[i][0];
	            dy=posY-ruta[i][1];
	            double dist=Math.sqrt(dx*dx+dy*dy);
	            if(dist<distMin) {
	                indMin=i;
	                distMin=dist;
	            }
	            
	        }
	        return indMin;
	}

	/**
	 * Indice del punto más cercano de ruta al punto
	 * @param ruta array con los puntos de la tractoria
	 * @param pos punto a buscar
	 * @return Índice de la ruta en el que se encuentra el punto de la 
	 * ruta más cercano al punto pos
	 */
	public static int indiceMasCercano(double[][] ruta,double pos[]){
		return indiceMasCercano(ruta, pos[0], pos[1]);
	}

	/**
	 * Método optimizado de búsqueda del punto más cercano utilizando 
	 * la información del último punto más cercano. Si en el parámetro indMinAnt se pasa
	 * un número negativo realiza una búsqueda exaustiva
	 * @param ruta array con los puntos de la tractoria
	 * @param esCerrada si la ruta debe considerarse cerrada
	 * @param posX Coordenada x del punto
	 * @param posY Coordenada y del punto
	 * @param indMinAnt indice donde estaba el mínimo en la iteración anterior
	 * @return Índice de la ruta en el que se encuentra el punto de la 
	 * ruta más cercano al punto pos
	 */
	public static int indiceMasCercanoOptimizado(double[][] ruta, boolean esCerrada,double posX,double posY,int indMinAnt){        
	    if(indMinAnt<0){
	    	return indiceMasCercano(ruta, posX, posY);
	    }
	    double dx;
	    double dy;
	    double distMin=Double.POSITIVE_INFINITY;
	    int indMin=0;
	    int indiceInicial = indMinAnt - 10;
	    if (esCerrada){
	    	indiceInicial = (indMinAnt + ruta.length - 10)%ruta.length;
	    }else{        	
	    	if (indiceInicial <= 0)
	            indiceInicial = 0;
	    }        
	    boolean encontrado=false;
		for(int i=indiceInicial;encontrado!=true; i=(i+1)%ruta.length) {
	            dx=posX-ruta[i][0];
	            dy=posY-ruta[i][1];
	            double dist=Math.sqrt(dx*dx+dy*dy);                
	            if(dist<=distMin) {
	                indMin=i;
	                distMin=dist;                   
	            }else{                    
	                encontrado=true;
	            }   
	    }
	    return indMin;
	}

	/**
	 * Método optimizado de búsqueda del punto más cercano utilizando 
	 * la información del último punto más cercano. Se busca entorno a ese.
	 * Si en el parámetro indMinAnt se pasa un número negativo realiza una búsqueda exaustiva
	 * @param ruta array con los puntos de la tractoria
	 * @param esCerrada si la ruta debe considerarse cerrada
	 * @param pos punto a buscar
	 * @param indMinAnt indice donde estaba el mínimo en la iteración anterior. Si es negativo no se tiene en cuenta
	 * @return Índice de la ruta en el que se encuentra el punto de la 
	 * ruta más cercano al punto pos
	 */
	public static int indiceMasCercanoOptimizado(double[][] ruta, boolean esCerrada,double[] pos,int indMinAnt){
		if(pos==null)
			throw new IllegalArgumentException("Vector de posición pasado es NULL");
		return indiceMasCercanoOptimizado(ruta, esCerrada, pos[0], pos[1], indMinAnt);
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
}
