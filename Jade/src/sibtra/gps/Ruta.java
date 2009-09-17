/**
 * 
 */
package sibtra.gps;

import java.io.Serializable;
import java.util.Vector;

import sibtra.util.UtilCalculos;
import Jama.Matrix;

/**
 * Contendrá una ruta grabada por el GPS
 * @author alberto
 *
 */
public class Ruta implements Serializable {

	/**
	 * Número de serie. IMPORTANTE porque vamos a salvarlo en fichero directamente.
	 * Si cambiamos estructura del objeto tenemos que cambiar el número de serie y ver 
	 * como se cargan versiones anteriores.
	 * Para saber si es necesario cambiar el número ver 
	 *  http://java.sun.com/j2se/1.5.0/docs/guide/serialization/spec/version.html#9419
	 */
	private static final long serialVersionUID = 3L;

	/** contendrá los puntos de la ruta */
	Vector<GPSData> puntos;

	/** Punto mas cercano al centro de la ruta */
	GPSData centro = null;

	/** Número máximo de puntos que contendrá la ruta */
	int tamMaximo;

	/** Si la ruta se considera espacial */
	boolean esEspacial;

	/** minima distancia que quienten que estar separados dos puntos para que se consideren distintos
	 * y se almacenen en el buffer espacial.
	 */
	double minDistOperativa = 0.7;

	/** Matriz de cambio de coordenadas ECEF a coordenadas locales */
	Matrix T = null;

	/** Deviación magnética calculada */
	double desviacionM=Double.NaN;

	/** umbral de angulo utilizado para determinar punto en recta al calcular {@link #desviacionM} */
	double umbralDesviacion=Math.PI; //por defecto para versiones anteriores que no lo tienen

	/** umbral de angulo por defecto para determinar punto en recta al calcular {@link #desviacionM} */ 
	static final double umbralDesviacionDefecto = Math.toRadians(1);
	/** Vector de indices de puntos considerados al calcular la desviación magnética */
	Vector<Integer> indiceConsideradosDM=null;
	/** Vector con la deviación magnética de los puntos considerados */
	Vector<Double> dmI=null;
	/** Desviación magnética máxima en los puntos considerados */
	double dmMax;
	/** Desviación estandar de la desviación magnética media */
	double desEstDM;

	/** @return distancia cartesiana entre los dos puntos calculanda usando las coordenadas locales */
	public static double distEntrePuntos(GPSData ptoA,GPSData ptoB){
		//TODO usar la coordenada Z
		double dist=Double.POSITIVE_INFINITY;
		double dx = ptoB.getXLocal()-ptoA.getXLocal();
		double dy = ptoB.getYLocal()-ptoA.getYLocal();                    
		dist = Math.sqrt(dx*dx + dy*dy);
		return dist;
	}



	/** Constructor por defecto, no pone tamaño y supone que no es espacial */
	public Ruta() {
		puntos=new Vector<GPSData>();
		tamMaximo=Integer.MAX_VALUE;
		esEspacial=false;
	}

	/**
	 * Constructor en que se puede fijar tamaño
	 * @param tamañoMaximo número máximo de elementos
	 */
	public Ruta(int tamañoMaximo) {
		puntos=new Vector<GPSData>();
		tamMaximo=tamañoMaximo;
		esEspacial=false;
	}

	/**
	 * Constructor en que se dice si va a ser espcial
	 * @param esEspacial si es espacial
	 */
	public Ruta(boolean esEspacial) {
		puntos=new Vector<GPSData>();
		tamMaximo=Integer.MAX_VALUE;
		this.esEspacial=esEspacial;
	}

	/**
	 * Constructor completo
	 * @param tamañoMaximo número máximo de puntos
	 * @param esEspacial si es espacial o no
	 */
	public Ruta(int tamañoMaximo,boolean esEspacial) {
		puntos=new Vector<GPSData>();
		tamMaximo=tamañoMaximo;
		this.esEspacial=esEspacial;
	}

	/**
	 * Actualiza el {@link #centro} y la matriz de rotación {@link #T} que definen sistema de
	 * coordenadas locales.
	 * Utilizaremos como centro el punto de la ruta con mejor precisión (RMS)
	 */
	public void actualizaSistemaLocal () {
		GPSData centro=puntos.elementAt(0);
		double minRMS= centro.getRms();
		for (int i = 1; i < puntos.size(); i++) {			
			if (puntos.elementAt(i).getRms() < minRMS) {
				minRMS = puntos.elementAt(i).getRms();
				centro = puntos.elementAt(i);
			}            
		}
		actualizaSistemaLocal(centro);
	}

	/**
	 * Crea la matriz de rotación {@link #T} y fija el {@link #centro} usando el punto pasado
	 * @param ptoParaCentro punto que se usará como centro para definir el plano
	 */
	public void actualizaSistemaLocal(GPSData ptoParaCentro) {
		if(ptoParaCentro==null) return;
		centro=ptoParaCentro.calculaECEF();
		// Matriz de rotación en torno a un punto
		double v[][] = new double[3][];
		double lonCenRad=Math.toRadians(centro.getLongitud());
		double latCenRad=Math.toRadians(centro.getLatitud());
		v[0] = new double[] { -Math.sin(lonCenRad), Math.cos(lonCenRad), 0 };
		v[1] = new double[] { -Math.cos(lonCenRad) * Math.sin(latCenRad), -Math.sin(latCenRad) * Math.sin(lonCenRad), Math.cos(latCenRad) };
		v[2] = new double[] { Math.cos(latCenRad) * Math.cos(lonCenRad), Math.cos(latCenRad) * Math.sin(lonCenRad), Math.sin(latCenRad)};

		Matrix M1 = new Matrix(v);

		// Matriz de inversión del eje z en torno al eje x (Norte)
		double w[][] = new double[3][];
		w[0] = new double[] { 0, 1, 0 };
		w[1] = new double[] { -1, 0, 0 };
		w[2] = new double[] { 0, 0, 1 };
		Matrix M2 = new Matrix(w);

		T = M2.times(M1); 
	}

	/**
	 * Actualiza el {@link #centro} y la matriz {@link #T} usando los de la ruta pasada
	 * @param rutaUsar ruta de la que se tomará el centro y T
	 */
	public void actualizaSistemaLocal(Ruta rutaUsar) {
		if(rutaUsar==null || rutaUsar.T==null || rutaUsar.centro==null)
			return;
		T=rutaUsar.T;
		centro=rutaUsar.centro;
	}

	/**
	 * @return the t
	 */
	public Matrix getT() {
		return T;
	}

	/**
	 * Calcula y actualiza las coordenadas locales del punto pasado.
	 * Hace uso de la {@link #T las matriz de traslación}.
	 * Si no está definido el {@link #centro} o {@link #T} se inicializan a (-1,-1,-1) :-(
	 * @param pto punto a actualizar
	 */
	public GPSData setCoordenadasLocales(GPSData pto) {
		if (T == null || centro==null) {
			pto.setCoordLocal(null);
			return pto;
		}

		Matrix res = pto.getCoordECEF().minus(centro.getCoordECEF()); 
		res = T.times(res); //dejamos como vector columna
		pto.setCoordLocal(res);
		return pto;
	}

	/** actualiza coordenadas locales de todos los puntos de la ruta */
	public void actualizaCoordenadasLocales() {
		for(int i=0; i<puntos.size(); i++)
			setCoordenadasLocales(puntos.elementAt(i));
	}

	/** @return el número de puntos que hay en la ruta */
	public int getNumPuntos() {
		return puntos.size();
	}

	/** @return true si tiene sistema de coordenadas local ({@link #centro} y {@link #T}) */
	public boolean tieneSistemaLocal() {
		return T!=null && centro!=null;
	}

	/**
	 * Añade punto pasado a la ruta DUPLICANDOLO. Si esta es espacial sólo si esta a {@link #minDistOperativa} de 
	 * el último de la ruta.
	 * Se controla que el número de puntos no supere {@link #tamMaximo}.
	 * @param nuevodata punto a añadir.
	 * @return si se añadió (tiene sentido en los espaciales)
	 */
	public boolean add(GPSData nuevodata) {
		if(!esEspacial) {
			//para los buffers temporales siempre se añade quitando los primeros si no cabe
			GPSData data=new GPSData(nuevodata);
			//solo para el angulo
			if(puntos.size()>0) {
				GPSData ultpto=puntos.lastElement();
				ultpto.setAngulo(ultpto.calculaAnguloGPS(data));
			}
			else { data.setAngulo(0); }
			puntos.add(data);
			while(puntos.size()>tamMaximo) puntos.remove(0);
			return true;
		} else {
			//para los espaciales sólo se considera el punto si está suficientemente lejos de 
			// el último
			if(puntos.size()==0) {
				GPSData data=new GPSData(nuevodata);
				data.setAngulo(0); 
				//data.setVelocidad(0); 
				puntos.add(data);
				return true;
			} 
			if (puntos.lastElement().distancia(nuevodata)>minDistOperativa) {
				GPSData data=new GPSData(nuevodata);
				//puntos.lastElement().calculaAngSpeed(data);
				GPSData ultpto=puntos.lastElement();
				ultpto.setAngulo(ultpto.calculaAnguloGPS(data));
				puntos.add(data);
				while(puntos.size()>tamMaximo) puntos.remove(0);
				return true;
			}
			return false;
		}
	}

	/** @return último punto de la ruta, null si no hay ninguno */
	public GPSData getUltimoPto() {
		if(puntos.size()==0)
			return null;
		else
			return puntos.lastElement();
	}

	/**
	 * @param minDistOperativa nuevo valor para {@link #minDistOperativa}
	 */
	public void setMinDistOperativa(double minDistOperativa) {
		this.minDistOperativa = minDistOperativa;
	}

	/**
	 * @return valor de {@link #minDistOperativa}
	 */
	public double getMinDistOperativa() {
		return minDistOperativa;
	}

	/**
	 * @param i
	 * @return
	 */
	public GPSData getPunto(int i) {
		if(i>=puntos.size() || i<0)
			return null;
		else
			return puntos.elementAt(i);
	}
	/**
	 * 
	 * @return Devuelve el centro del sistema de coordenadas locales
	 */
	public GPSData getCentro() {
		return centro;
	}
	public String toString() {
		String retorno="Ruta "+(esEspacial?"ESPACIAL":"TEMPORAL")+" de "+puntos.size()+" puntos ";
		if(centro!=null)
			retorno+="CON sistema local centrado en "+centro;
		retorno+="\n";
		for(int i=0; i<puntos.size(); i++)
			retorno+=i+":"+puntos.get(i)+"\n";
		return retorno;
	}

	/** @return la desviación magnética {@link #desviacionM} con umbral de angulo por defecto. 
	 * La calcula si no está caclulada */
	public double getDesviacionM() {
		if(Double.isNaN(desviacionM) || umbralDesviacion!=umbralDesviacionDefecto)
			calculaDesM(umbralDesviacionDefecto);
		return desviacionM;
	}

	/** @return la desviación magnética {@link #desviacionM} con umbral de angulo pasado. 
	 * La calcula si no está caclulada */
	public double getDesviacionM(double umbralAngulo) {
		if(Double.isNaN(desviacionM) || umbralDesviacion!=umbralAngulo)
			calculaDesM(umbralAngulo);
		return desviacionM;
	}

	/** 
	 * Calcula la desviación magnetica comparando los datos de la IMU con los de la 
	 * evolución de la ruta obtenidos con el GPS.
	 * Se usarán sólo los tramos de ruta réctilíneos, para ello un punto sólo se tendrá en 
	 * cuenta si la diferencia de angulo con siguiente y anterior son es menor de un umbral
	 * @param umbralAngulo umbral para considerar punto en una recta. 
	 */
	private void calculaDesM(double umbralAngulo) {
//		double umbralAngulo=Math.toRadians(10); //umbral de angulos considerados
		if(puntos.size()<2)
			return; //no tocamos la desv.
		//vemos si tenemos datos IMU para todos
		for(int i=0; i<puntos.size(); i++) 
			if(puntos.get(i).getAngulosIMU()==null) {
				System.out.println("Punto "+i+" de la ruta no tienen angulos IMU. No podemos calcular");
				return;
			}
		//el angulo se calcula al añadir cada punto
		dmMax=0; //desviación máxima
		double dAcum=0; //desviación acumulada
		double dAcum2=0; //desviación acumulada al cuadrado
		if(indiceConsideradosDM==null)
			indiceConsideradosDM=new Vector<Integer>();
		else
			indiceConsideradosDM.clear();
		//desviación estandar de los considerados para no repetir en el cálculo desviación estandar 
		if(dmI==null)
			dmI=new Vector<Double>();
		else
			dmI.clear();
		for(int i=2; //el 0 no tiene angulo, por lo que el 1 tampoco se puede considerar
		i<(puntos.size()-1); //el último tampoco se puede considerar  
		i++) {
			double angI=puntos.get(i).getAngulo();
			double angI_1=puntos.get(i-1).getAngulo();
			double angIp1=puntos.get(i+1).getAngulo();
			if(Math.abs(UtilCalculos.normalizaAngulo(angI-angI_1))<=umbralAngulo
					&& Math.abs(UtilCalculos.normalizaAngulo(angI-angIp1))<=umbralAngulo ) {
				indiceConsideradosDM.add(i);
				double da=UtilCalculos.normalizaAngulo(puntos.get(i).getAngulo() 
						- Math.toRadians(puntos.get(i-1).getAngulosIMU().getYaw()));
				double daAbs=Math.abs(da);
				if(daAbs>dmMax) dmMax=daAbs;
				dAcum+=da;
				dAcum2+=da*da;
				dmI.add(da);
			}
		}

		desviacionM=dAcum/(indiceConsideradosDM.size()); //desviación media
		umbralDesviacion=umbralAngulo; //anotamos el umbral utilizado

		//desviación estandar de la desviación (valga la redundancia :-)
		double dif2=0.0;
		for(int i=0; i<dmI.size(); i++) 
			dif2+=(dmI.get(i)-desviacionM)*(dmI.get(i)-desviacionM);

		desEstDM=dif2/(indiceConsideradosDM.size());

//		System.out.println(" Con umbral="+Math.toDegrees(umbralAngulo)+" grados"
//				+" Considerados "+indiceConsideradosDM.size()+" de "+(puntos.size()-3)+" posibles"
//				+" Desviación media="+Math.toDegrees(desviacionM)
//				+" Desviación estandar="+Math.toDegrees(desEstDM)
//				+" Desviación máxima="+Math.toDegrees(dmMax));

	}

	/**
	 * Aplica {@link #distEntrePuntos(GPSData, GPSData)} a los puntos de indices pasados
	 * @param ind1 indice del primer punto considerado
	 * @param ind2 indice del segundo punto considerado 
	 * @return distancia cartesiana entre los dos puntos calculanda usando las coordenadas locales 
	 * */
	public double distEntrePuntos(int ind1, int ind2){
		return distEntrePuntos(getPunto(ind1), getPunto(ind2));
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {

	}
}
