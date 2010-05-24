/**
 * 
 */
package sibtra.gps;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.StringTokenizer;

import sibtra.imu.AngulosIMU;
import sibtra.util.UtilCalculos;

/**
 * Representa trayectorias que puede seguir verdino en coordenadas locales.
 * 
 * @author alberto
 *
 */
public class Trayectoria implements Serializable {

	/**
	 * Número de serie. IMPORTANTE porque vamos a salvarlo en fichero directamente.
	 * Si cambiamos estructura del objeto tenemos que cambiar el número de serie y ver 
	 * como se cargan versiones anteriores.
	 * Para saber si es necesario cambiar el número ver 
	 *  http://java.sun.com/j2se/1.5.0/docs/guide/serialization/spec/version.html#9419
	 */
 	private static final long serialVersionUID = 1L;
	
	private static final double alejamientoRazonable = 0;
	boolean esCerrada=false;
	double distanciaMaxima=Double.MAX_VALUE;
	
	/** indice del más cercano encontrado en la ultima iteracion*/
	int indiceUltimoCercano=-1;

	/** ultima posicion del coche pasada */
	double[] ultimaPosCoche=null;
	
	/** Longitud de la trayectoria */
	protected double largo=-1.0;
	
	public double[] x=null;
	public double[] y=null;
	public double[] z=null;
	public double[] rumbo=null;
	public double[] velocidad=null;
	
	public String toString() {
		String res=String.format(Locale.US,"Tr {num=%d,%s,l=%f m}", length()
				,esCerrada?"cerrada":"abierta"
			, getLargo());
		for(int i=0;i<length();i++)
			res+=String.format(Locale.US,"\n(%f , %f , %f) %fº %fm/s",x[i],y[i],z[i]
			      ,Math.toDegrees(rumbo[i]),velocidad[i]);
		return res;
	}
	
	/** copia todos los campos de trOriginal sobre los de la trayectoria actual */
	protected void copia(Trayectoria trOriginal) {
		esCerrada=trOriginal.esCerrada;
		distanciaMaxima=trOriginal.distanciaMaxima;
		
		x=new double[trOriginal.x.length];
		System.arraycopy(trOriginal.x,0,x,0,trOriginal.x.length);
		y=new double[trOriginal.y.length];
		System.arraycopy(trOriginal.y,0,y,0,trOriginal.y.length);
		z=new double[trOriginal.z.length];
		System.arraycopy(trOriginal.z,0,z,0,trOriginal.z.length);
		rumbo=new double[trOriginal.rumbo.length];
		System.arraycopy(trOriginal.rumbo,0,rumbo,0,trOriginal.rumbo.length);
		velocidad=new double[trOriginal.velocidad.length];
		System.arraycopy(trOriginal.velocidad,0,velocidad,0,trOriginal.velocidad.length);
		
		
	}

	/** Invoca a {@link #Trayectoria(Ruta, double, double, double)} tomando la desviación magnética de la que
	 * se calcula a partir de los  puntos de la ruta.
	 * @param ruta de donde tomar los puntos
	 * @param nuevaDistMax distancia máxima que se quiere que exista entre los puntos
	 * @param umbral para considerar la ruta como cerrada
	 */
	public Trayectoria(Ruta ruta, double nuevaDistMax, double umbral) {
		this(ruta,nuevaDistMax,umbral,ruta.getDesviacionM());
	}
	
	/** Constructor a partir de una ruta.
	 * @param ruta de donde tomar los puntos
	 * @param nuevaDistMax distancia máxima que se quiere que exista entre los puntos, 
	 * 	se usa al invocando a {@link #nuevaDistanciaMaxima(double)} 
	 * @param umbral para considerar la ruta como cerrada
	 * @param desvMagnética declinación magnética a aplicar
	 */
	public Trayectoria(Ruta ruta, double nuevaDistMax, double umbral, double desvMagnética) {
		if(ruta==null)
			throw new IllegalArgumentException("La ruta pasada para construir la trayectoria no puede ser null");
		distanciaMaxima=0.0;
		int indUltimo=indiceUltimoConsiderar(ruta, umbral);
		x=new double[indUltimo+1];
		y=new double[indUltimo+1];
		z=new double[indUltimo+1];
		rumbo=new double[indUltimo+1];
		velocidad=new double[indUltimo+1];
		if(indUltimo==-1)  //si la ruta no tienen puntos
			return;
		GPSData ptoA;
		GPSData ptoB = ruta.getPunto(0);                
		AngulosIMU aiA;
		for(int i=0; i<=(indUltimo-1); i++) {
			ptoA=ptoB;
			ptoB=ruta.getPunto(i+1);
			x[i]=ptoA.getXLocal();
			y[i]=ptoA.getYLocal();
			z[i]=ptoA.getZLocal();
			aiA = ptoA.getAngulosIMU();
			rumbo[i]=((aiA != null) ? Math.toRadians(aiA.getYaw()) : ptoA.calculaAnguloGPS(ptoB))
					+ desvMagnética;
			velocidad[i]= (ptoA.getVelocidad()!=Double.NaN)? ptoA.getVelocidad() : ptoA.calculaVelocidadGPS(ptoB);
			//vamos actualizando la distancia maxima
			double distAB=Ruta.distEntrePuntos(ptoA, ptoB);
			if(distAB>distanciaMaxima) distanciaMaxima=distAB;
		}
		//Para el último punto
		ptoA=ptoB;
		ptoB=ruta.getPunto(0); //por si es cerrada
		x[indUltimo]=ptoA.getXLocal();
		y[indUltimo]=ptoA.getYLocal();
		z[indUltimo]=ptoA.getZLocal();
		aiA = ptoA.getAngulosIMU();
		rumbo[indUltimo]=(
				(aiA != null) ? Math.toRadians(aiA.getYaw()) //si hay ángulos el Yaw
					:(esCerrada?ptoA.calculaAnguloGPS(ptoB) //si es cerrada se calcula con el primer punto
							:(indUltimo>0?rumbo[indUltimo-1]:Double.NaN)) //se toma el del punto anterior (si lo hay)
		)+ desvMagnética;
		velocidad[indUltimo]= (
				ptoA.getVelocidad()!=Double.NaN? ptoA.getVelocidad()  //si tiene velocidad se toma
						: (esCerrada?ptoA.calculaVelocidadGPS(ptoB) //si es cerrada se calcula con 1º
								: velocidad[indUltimo-1]) //se usa la misma que en el punto anterior
						);
		//para la distancia maxima.
		if(esCerrada) {
			double distAB=Ruta.distEntrePuntos(ptoA, ptoB);
			if(distAB>distanciaMaxima) distanciaMaxima=distAB;
		}
		nuevaDistanciaMaxima(nuevaDistMax);
	}
	
	/** Se invoca a {@link #Trayectoria(Ruta, double, double)} 
	 * Se usa 3.0 metros para considerar si es cerrada
	 * @param ruta de donde tomar los puntos 
	 * @param nuevaDistMax distancia máxima que se quiere que exista entre los puntos
	 */
	public Trayectoria(Ruta ruta, double nuevaDistMax) {
		this(ruta,nuevaDistMax,3.0);		
	}
	
	/** Se invoca a {@link #Trayectoria(Ruta, double)} 
	 * No añade puntos ya que utiliza {@link Double.#MAX_VALUE} para la distancia maxima entre puntos
	 * @param ruta de donde tomar los puntos
	 */
	public Trayectoria(Ruta ruta) {
		this(ruta,Double.MAX_VALUE,3.0);
	}
	
	/** Constructor de copia */
	public Trayectoria(Trayectoria tr) {
		this(tr,Double.MAX_VALUE);
		
	}
	
	/** Constructor a partir de trayectoria pero con otra separación minima */
	public Trayectoria(Trayectoria tr, double distMax) {
		copia(tr);
		nuevaDistanciaMaxima(distMax);
	}
	
	/** Constructor en el que los datos se pasan en array bidimensional de doubles.
	 * Deben estar al menos (x,y) pero pueden estar (x,y,z,rumbo,velocidad).
	 * Por defecto (x,y,0, calculado de un punto a otro, 1)
	 * @param puntos
	 */
	public Trayectoria(double[][] puntos) {
		if(puntos==null || puntos.length==0)
			throw new IllegalArgumentException("Para construir trayectoria, vector tiene que tener puntos");
		if ( puntos[0].length<2)
			throw new IllegalArgumentException("Deben haber coordenadas x e y al menos");
		distanciaMaxima=0.0;
		int largo=puntos.length;
		x=new double[largo];
		y=new double[largo];
		z=new double[largo];
		velocidad=new double[largo];
		rumbo=new double[largo];
		int i=0;
		for(int cont=0;cont<puntos.length-1;cont++) {
			x[i]=puntos[i][0];
			y[i]=puntos[i][1];
			z[i]=(puntos[i].length>=3)?puntos[i][2]:0.0;
			rumbo[i]=(puntos[i].length>=4)?puntos[i][3]:-Math.atan2(puntos[i+1][1]-puntos[i][1], puntos[i+1][0]-puntos[i][0]);
			velocidad[i]=(puntos[i].length>=5)?puntos[i][4]:1.0;
			//vamos actualizando la distancia maxima
			double distAB=UtilCalculos.distanciaPuntos(puntos[i], puntos[i+1]);
			if(distAB>distanciaMaxima) distanciaMaxima=distAB;
			i++;
		}
		//punto final
		x[i]=puntos[i][0];
		y[i]=puntos[i][1];
		z[i]=(puntos[i].length>3)?puntos[i][2]:0.0;
		rumbo[i]=(puntos[i].length>4)?puntos[i][3]:rumbo[i-1];  //si no hay usamos rumbo del anterior
		velocidad[i]=(puntos[i].length>5)?puntos[i][4]:1.0;
		
		esCerrada=false;
	}
	
	public Trayectoria (double[] puntoInicial, double[] puntoFinal, double distMax,double vel){
		if(puntoInicial==null || puntoFinal==null ||puntoInicial.length==0 || puntoFinal.length==0)
			throw new IllegalArgumentException("Para construir trayectoria se necesitan los puntos inicial y final");
		if ( puntoInicial.length<2 || puntoFinal.length<2)
			throw new IllegalArgumentException("Los puntos inicial y final tienen que tener al menos coordenadas (x,y)");		
		double dx = puntoFinal[0] - puntoInicial[0];
		double dy = puntoFinal[1] - puntoInicial[1];
		double dz = ((puntoFinal.length>3)?puntoFinal[2]:0.0) - ((puntoInicial.length>3)?puntoInicial[2]:0.0);
		double zini=((puntoFinal.length>3)?puntoFinal[2]:0.0);
		double distEntrePtos = Math.sqrt(dx*dx +dy*dy);
		double angEntrePtos = Math.PI-Math.atan2(dy,dx);
		int numPuntos = (int)Math.ceil(distEntrePtos/distMax);
		double incX = dx/numPuntos;
		double incY = dy/numPuntos;
		double incZ = dz/numPuntos;
		x=new double[numPuntos];
		y=new double[numPuntos];
		z=new double[numPuntos];
		velocidad=new double[numPuntos];
		rumbo=new double[numPuntos];
		for (int i=0;i<numPuntos;i++){
			x[i] = puntoInicial[0] + incX*i;
			y[i] = puntoInicial[1] + incY*i;
			z[i] = zini + incZ*i;
			rumbo[i] = angEntrePtos;
			velocidad[i] = vel ; 
		}
		esCerrada = false;
	}
	
	/** Cambia la distancia máxima añadiendo puntos si es necesario.
	 * Es privado. Si se quiera trayectoria con otra distancia maxima, usar constructor.
	 * @param distMax
	 */
	protected void nuevaDistanciaMaxima(double distMax) {
		if(distMax>=distanciaMaxima)
			return; //ya lo tenemos, no hay que hacer nada
		ArrayList<Double> xR=new ArrayList<Double>();
		ArrayList<Double> yR=new ArrayList<Double>();
		ArrayList<Double> zR=new ArrayList<Double>();
		ArrayList<Double> rumboR=new ArrayList<Double>();
		ArrayList<Double> velocidadR=new ArrayList<Double>();
		for(int i=0; i<x.length-(esCerrada?0:1); i++) {
			xR.add(x[i]);
			yR.add(y[i]);
			zR.add(z[i]);
			rumboR.add(rumbo[i]);
			velocidadR.add(velocidad[i]);
			int isig=(i+1)%x.length;  //indice del siguiente punto
			double dx=x[isig]-x[i];
			double dy=y[isig]-y[i];
			double dz=z[isig]-z[i];
			double separacion=Math.sqrt(dx*dx + dy*dy+ dz*dz);
			//hay que rellenar con puntos intermedios (o no si sale 0)
			int numPuntosIntermedios=(int) Math.ceil(separacion / distMax)-1;
			double drumbo = UtilCalculos.normalizaAngulo(rumbo[isig] - rumbo[i]);  //variación en angulo  en el tramo
			double dvelocidad = velocidad[isig]-velocidad[i];  //variacion en velocidad en el tramo
			if(numPuntosIntermedios>0) distanciaMaxima=distMax; //la distancia maxima es la solicitada
			for (int k = 1; k <= numPuntosIntermedios; k++) {
				//Tendremos (numPuntosIntermedios+1) tramos intermedios.
				//Por eso dividimos todas las magnitudes por (numPuntosIntermedios+1)
				xR.add(x[i]+k*dx/(numPuntosIntermedios+1));
				yR.add(y[i]+k*dy/(numPuntosIntermedios+1));
				zR.add(z[i]+k*dz/(numPuntosIntermedios+1));
				rumboR.add(rumbo[i]+k*drumbo/(numPuntosIntermedios+1));
				velocidadR.add(velocidad[i]+k*dvelocidad/(numPuntosIntermedios+1));
			}
		}
		if(!esCerrada) {
			//tenemos que añadir el último punto.
			int i=x.length-1;
			xR.add(x[i]);
			yR.add(y[i]);
			zR.add(z[i]);
			rumboR.add(rumbo[i]);
			velocidadR.add(velocidad[i]);
		}
		//convertimos las listas a array
		x=listaAArray(xR);
		y=listaAArray(yR);
		z=listaAArray(zR);
		rumbo=listaAArray(rumboR);
		velocidad=listaAArray(velocidadR);
	}

	/** @return array a partir de lista */
	protected double[] listaAArray(List<Double> li ) {
		double[] res=new double[li.size()];
		for(int i=0; i<li.size(); i++)
			res[i]=li.get(i);
		return res;
	}
	
	
	/**
	 * Nos dice si la ruta pasada es cerrad y devuelve (en cualquier caso) el indice de último punto a considerar.
	 * Método que mide la distancia entre el último punto de la ruta y el primero
	 * para decidir si la ruta está cerrada o no. Debido a que el principio de la ruta
	 * y el final pueden estar solapados, no solo hay que comprobar la distancia 
	 * entre el último punto y el primero, si no también con los puntos siguientes
	 * al primero para ver con cual se da la distancia mínima. Cuando ya tenemos la 
	 * distMin se comprueba si el valor está por debajo del umbral que indica si ambos
	 * puntos están lo sufientemente cerca como para suponer que la ruta está cerrada 
	 *  
	 *  @param ruta donde sacar los puntos
	 *  @param umbral distancia para considerar ruta cerrada.
	 * @return indice del ultimo punto de la ruta a tener en cuenta. Si es abierta será el índice del último pto de la ruta.
	 */
	protected int indiceUltimoConsiderar(Ruta ruta, double umbral){
		int indiceFinal = ruta.getNumPuntos()-1;
		esCerrada=false;
		//necesitamos al menos 2 puntos
		if(ruta.getNumPuntos()<2)
			return indiceFinal;
		double distMin = Double.POSITIVE_INFINITY;
		GPSData ptoInicial = ruta.getPunto(0);
		GPSData ptoAux = ruta.getPunto(ruta.getNumPuntos()-1);
		double distAux = Ruta.distEntrePuntos(ptoInicial,ptoAux);
		int i = ruta.getNumPuntos()-1;
		int indiceAux=indiceFinal;
		//buscamos en puntos anteriores mientras se acerquen al punto inicial
		while ((distAux <= distMin) && (i>=1)){
			distMin = distAux;
			indiceAux = i;
			i--;
			ptoAux = ruta.getPunto(i);
			distAux = Ruta.distEntrePuntos(ptoInicial,ptoAux);        		
		}
		if (indiceAux==1 || distMin>=umbral) {
			//Es abierta
			esCerrada = false;
			return ruta.getNumPuntos()-1;
		} 
		//puntos por debajo del umbral
		esCerrada = true;
		boolean encontrado = false;
		indiceFinal = indiceAux;
		//seguimos para atrás hasta que no haya un salto muy brusco de rumbo
		for (int j=indiceFinal;encontrado!=true;j--){
			GPSData ptoA = ruta.getPunto(j);
			GPSData ptoAnt = ruta.getPunto(j-1);
			double anguloFinal = ptoA.calculaAnguloGPS(ptoInicial);
			double anguloPtoA = ptoAnt.calculaAnguloGPS(ptoA);
			if (Math.abs(anguloFinal-anguloPtoA)<Math.PI/6){
				encontrado = true;
				indiceFinal = j;
			}
		}
		return indiceFinal;
	}

	/**
	 * Indice del punto más cercano de trayectoria al punto pasado.
	 * Se hace búsqueda de fuerza bruta (comparando con todos los puntos de la trayectoria)
	 * @param posX Coordenada x del punto
	 * @param posY Coordenada y del punto
	 * @return Índice de la ruta en el que se encuentra el punto de la 
	 * ruta más cercano al punto (posX,posY)
	 */
	public int indiceMasCercano(double posX,double posY){
		//Buscar punto más cercano al coche
		double distMin=Double.POSITIVE_INFINITY;
		int indMin=0;
		double dx;
		double dy;
		for(int i=0; i<x.length; i++) {
			dx=posX-x[i];
			dy=posY-y[i];
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
	 * @param pos punto a buscar
	 * @return Índice de la ruta en el que se encuentra el punto de la 
	 * ruta más cercano al punto pos
	 */
	public int indiceMasCercano(double pos[]){
		return indiceMasCercano( pos[0], pos[1]);
	}

	/**
	 * Método optimizado de búsqueda del punto más cercano utilizando 
	 * la información del último punto más cercano. Si en el parámetro indMinAnt se pasa
	 * un número negativo realiza una búsqueda exaustiva
	 * @param posX Coordenada x del punto
	 * @param posY Coordenada y del punto
	 * @param indMinAnt indice donde estaba el mínimo en la iteración anterior
	 * @return Índice de la ruta en el que se encuentra el punto de la 
	 * ruta más cercano al punto pos
	 */
	public int indiceMasCercanoOptimizado(double posX,double posY,int indMinAnt){
		if(length()==1)
			return 0;
	    if(indMinAnt<0){
	    	return indiceMasCercano(posX, posY);
	    }
	    double dx;
	    double dy;
	    double distMin=Double.POSITIVE_INFINITY;
	    int indMin=0;
	    int indiceInicial = indMinAnt - 10;
	    if (esCerrada){
	    	indiceInicial = (indMinAnt + length() - 10)%length();
	    }else{        	
	    	if (indiceInicial <= 0)
	            indiceInicial = 0;
	    }        
	    boolean encontrado=false;
		for(int i=indiceInicial;encontrado!=true; i=(i+1)%length()) {
	            dx=posX-x[i];
	            dy=posY-y[i];
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
	 * @param pos punto a buscar
	 * @param indMinAnt indice donde estaba el mínimo en la iteración anterior. Si es negativo no se tiene en cuenta
	 * @return Índice de la ruta en el que se encuentra el punto de la 
	 * ruta más cercano al punto pos
	 */
	public int indiceMasCercanoOptimizado(double[] pos,int indMinAnt){
		if(pos==null)
			throw new IllegalArgumentException("Vector de posición pasado es NULL");
		return indiceMasCercanoOptimizado(pos[0], pos[1], indMinAnt);
	}

	/** @return número de puntos de la trayectoria */
	public int length() {
		return x.length;
	}

//	/** @return array con las coordenadas de dos dimesiones con las coordenadas X e Y*/
//	public double[][] getArrayXY() {
//		double[][] arraXY=new double[x.length][2];
//		for(int i=0;i<x.length; i++) {
//			arraXY[i][0]=x[i];
//			arraXY[i][1]=y[i];
//		}			
//		return arraXY;
//	}
	
	public boolean esCerrada() {
		return esCerrada;
	}

	public void situaCoche(double x, double y){
		double[] posCoche={x,y};
		situaCoche(posCoche);
	}
	
	public void situaCoche(double[] posCoche) {
		if(posCoche==null || posCoche.length<2)
			throw new IllegalArgumentException("La posición de coche debe ser array de al menos 2 valores (x,y)");
		if(ultimaPosCoche==null 
				|| UtilCalculos.distanciaPuntos(posCoche, ultimaPosCoche)> alejamientoRazonable
				|| indiceUltimoCercano<0 ) {
			//no tenemos posición anterior, está muy lejos o no hay indice anterior
			indiceUltimoCercano=indiceMasCercano(posCoche);
			ultimaPosCoche=posCoche;			
		} else {
			//aporvechamos el indice anterior
			indiceUltimoCercano=indiceMasCercanoOptimizado(posCoche, indiceUltimoCercano);
		}
	}
	
	/** @return el indice del más cercano. Para invocar después de {@link #situaCoche(double[])} */
	public int indiceMasCercano() {
		if(indiceUltimoCercano<0 || ultimaPosCoche==null)
			throw new IllegalStateException("La posción de el coche aún no se ha establecido");
		return indiceUltimoCercano;
	}
	
	/** @return si hay fijada posición de coche */
	public boolean hayPosicionCoche() {
		return indiceUltimoCercano>=0 && ultimaPosCoche!=null;
	}
	
	/** @return la distancia entre la ultima posición establecida del coche con {@link #situaCoche(double[])} y
	 * el punto más cercano de la trayectoria
	 */
	public double distanciaAlMasCercano() {
		if(indiceUltimoCercano<0 || ultimaPosCoche==null)
			throw new IllegalStateException("La posción de el coche aún no se ha establecido");
		double[] pto={x[indiceUltimoCercano],y[indiceUltimoCercano]};
		return UtilCalculos.distanciaPuntos(pto, ultimaPosCoche);
	}
	
	/**
	 * Calcula largo de la trayectoria entre indIni y punto anterior al indFin 
	 * @return el largo de la trayectoria entre los puntos de los indices pasados */
	public double getLargo(int indIni,int indFin) {
		if(indIni<0 || indIni>x.length || indFin<0 || indFin>x.length)
			throw new IllegalArgumentException("Indices fuera de rango");
		if(!esCerrada && indIni>indFin )
			throw new IllegalArgumentException("Indice inicial el mayor que final (y no es cerrada)");
    	double dist = 0;
    	int indFinal = indFin;
    	if (indIni == indFin-1){//Para que entre en el bucle en caso de índices consecutivos
    		indFinal = indFinal + 1;
    	}//La condición de parada no funciona si indIni y indFin son consecutivos,no entra en el bucle
    	for (int i=indIni;i!=(indFinal+x.length-1)%x.length;i++){
    		double dx=x[i]-x[(i+1)%x.length];
    		double dy=y[i]-y[(i+1)%x.length];
    		dist += Math.sqrt(dx*dx+dy*dy);
    	}    	
    	return dist;
	}
	
	/** @param distancia distancia mínima que debe separar los puntos
	 * @param indIni indice del punto a partir del cual se empieza a medir 
	 * @return el indice del punto que esta a la distancia pasada del inicial 
	 */
	public int indiceHastaLargo(double distancia, int indIni) {
		if(indIni<0 || indIni>x.length )
			throw new IllegalArgumentException("Indice fuera de rango");
    	double dist = 0;
    	int i = 0;    	
    	for (i=indIni;dist<distancia && (esCerrada||i<(length()-1));i++){
            dist += getLargo(i,(i+1)%length());
    	}
    	return i+1;
	}

	/** @return la última posición establecida con {@link #situaCoche(double[])} */
	public double[] getPosicionCoche() {
		return ultimaPosCoche;
	}

	/** @return la longitud total de la trayectoria */
	public double getLargo() {
		if(largo<0)
			if(esCerrada)
				largo=getLargo(0, 0);
			else
				largo=getLargo(0, length());
		return largo;
	}

	/**
	 * Salva la trayecetoria actual en el fichero indicado
	 * @param nombreFichero fichero en el que salvar la trayectoria
	 * @return si se salvó correctamente
	 */
	public boolean salvaAFichero(String nombreFichero)  {
		if (nombreFichero==null || nombreFichero.length()==0)
			throw new IllegalArgumentException("Necesario nombre de fichero ");
		try {
			File file = new File(nombreFichero);
			ObjectOutputStream oos;
			oos = new ObjectOutputStream(new FileOutputStream(file));
			oos.writeObject(this);
			oos.close();
			return true;
		} catch (Exception e) {
			System.err.println("Problema al salvar trayectoria en fichero "+nombreFichero);
			return false;
		}
	}

	/**
	 * Carga la trayecetoria del fichero indicado
	 * @param nombreFichero fichero del que cargar la trayectoria
	 * @return trayectoria cargada o null si hay error
	 */
	public static Trayectoria cargaDeFichero(String nombreFichero)  {
		if (nombreFichero==null || nombreFichero.length()==0)
			throw new IllegalArgumentException("Necesario nombre de fichero ");
		try {
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(nombreFichero));
			Trayectoria trayectoria=(Trayectoria)ois.readObject();
			ois.close();
			return trayectoria;
		} catch (Exception e) {
			System.err.println("Problema al cargar trayectoria del fichero "+nombreFichero);
			return null;
		}
	}

//	/**
//	 * Salva la trayecetoria actual en el fichero indicado
//	 * @param nombreFichero fichero en el que salvar la trayectoria
//	 * @return si se salvó correctamente
//	 */
//	public boolean salvaAFicheroOct(String nombreFichero)  {
//		if (nombreFichero==null || nombreFichero.length()==0)
//			throw new IllegalArgumentException("Necesario nombre de fichero ");
//		try {
//			FileWriter fileW = new FileWriter(nombreFichero);
//			//primera linea
//			fileW.write("# Created by "+getClass().getName()+", "+System.currentTimeMillis());
//			//TODO Hacer clase para salvar en octave?? usar MAT
//			return true;
//		} catch (Exception e) {
//			System.err.println("Problema al cargar trayectoria del fichero "+nombreFichero);
//			return false;
//		}
//	}

	/**
	 * Carga matriz Tr de fichero octave y genera nueva trayectoria con los datos
	 * @param nombreFichero fichero donde cargarla
	 * @return la nueva trayectoria o null si hay algún problema
	 */
	public static Trayectoria cargaDeFicheroOctave(String nombreFichero)  {
		if (nombreFichero==null || nombreFichero.length()==0)
			throw new IllegalArgumentException("Necesario nombre de fichero ");
		Trayectoria nuevaTr=null;
		double[][] datTr=null;
		try {
			BufferedReader fent=new BufferedReader( new FileReader(nombreFichero));
			//Cabecera
			String lin=fent.readLine();
			//nombre
			while((lin=fent.readLine())!=null) {
				StringTokenizer st=new StringTokenizer(lin);
				st.nextToken(); //#
				if(!"name:".equals(st.nextToken())) throw null;
				String nombre=st.nextToken();
				//tipo
				lin=fent.readLine();
				st=new StringTokenizer(lin);
				st.nextToken(); //#
				if(!"type:".equals(st.nextToken())) throw null;
				String tipo=st.nextToken();
				if("scalar".equals(tipo)) {
					fent.readLine();
					continue;
				}
				if(!"matrix".equals(tipo)) throw null;
				//filas
				lin=fent.readLine();
				st=new StringTokenizer(lin);
				st.nextToken(); //#
				if(!"rows:".equals(st.nextToken())) throw null;
				int filas=Integer.valueOf(st.nextToken());
				//columnas
				lin=fent.readLine();
				st=new StringTokenizer(lin);
				st.nextToken(); //#
				if(!"columns:".equals(st.nextToken())) throw null;
				int columnas=Integer.valueOf(st.nextToken());
				if(!"Tr".equals(nombre)) {
					//quitamos todas las filas
					for(int i=1;i<=filas;i++)
						fent.readLine();
					continue;  //probamos siguiente matriz
				} 
				//leemos la matriz
				datTr=new double[filas][columnas];
				for(int i=0; i<filas; i++) {
					lin=fent.readLine();
					st=new StringTokenizer(lin);
					for(int j=0;j<columnas;j++)
						datTr[i][j]=Double.valueOf(st.nextToken());
				}
				nuevaTr=new Trayectoria(datTr);
			}
			return nuevaTr;
		} catch (Exception e) {
			return null;
		}
	}
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {

		String nombreFichero="Trayectorias/pru1";
		int nptos=20;
		double[][] dtos=new double[nptos][];
		for(int i=0;i<nptos;i++) {
			double[] da={i,i};
			dtos[i]=da;
		}
		Trayectoria tr=new Trayectoria(dtos);
		
		tr.salvaAFichero(nombreFichero);
		
		Trayectoria tc=Trayectoria.cargaDeFichero(nombreFichero);
		System.out.println("Trayectoria leida:"+tc);
		
		Trayectoria tcot=Trayectoria.cargaDeFicheroOctave("Trayectorias/prueba.oct");
		System.out.println("Trayectoria leida:"+tcot);
		
	}

}
