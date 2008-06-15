/**
 * 
 */
package sibtra.gps;

import java.io.Serializable;
import java.util.Vector;

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
	 */
	private static final long serialVersionUID = 2L;

	/** contendrá los puntos de la ruta */
	Vector<GPSData> puntos;
	
	/** Punto mas cercano al centro de la ruta */
	private GPSData centro = null;

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
	 */
	public void actualizaSistemaLocal () {
		GPSData centro;
		if ( puntos.size() == 0)
			return;                

		// Primero buscamos el punto central exacto
		Matrix central=new Matrix(3,1);  //se inicializa a 0
		for (int i = 0; i < puntos.size(); i++) {
			central.plusEquals(puntos.elementAt(i).getCoordECEF());
		}
		central.timesEquals(1/(double)puntos.size());

		// Ahora buscamos el punto que más cerca esté de ese centro
		double dist = central.minus(puntos.elementAt(0).getCoordECEF()).normF(); 
		centro = puntos.elementAt(0);
		for (int i = 0; i < puntos.size(); i++) {
			double myDist = central.minus(puntos.elementAt(i).getCoordECEF()).normF(); 
			if (myDist < dist) {
				dist = myDist;
				centro = puntos.elementAt(i);
			}            
		}
		actualizaSistemaLocal(centro);
	}
	
	/**
	 * Crea la matriz de rotación {@link #T} y fija el {@link #centro} usando el punto pasado
	 * @param centro punto que se usará como centro para definir el plano
	 */
	public void actualizaSistemaLocal(GPSData ptoParaCentro) {
		if(ptoParaCentro==null) return;
		centro=ptoParaCentro;
		// Matriz de rotación en torno a un punto
		double v[][] = new double[3][];
		v[0] = new double[] { -Math.sin(centro.getLongitud()), Math.cos(centro.getLongitud()), 0 };
		v[1] = new double[] { -Math.cos(centro.getLongitud()) * Math.sin(centro.getLatitud()), -Math.sin(centro.getLatitud()) * Math.sin(centro.getLongitud()), Math.cos(centro.getLatitud()) };
		v[2] = new double[] { Math.cos(centro.getLatitud()) * Math.cos(centro.getLongitud()), Math.cos(centro.getLatitud()) * Math.sin(centro.getLongitud()), Math.sin(centro.getLatitud())};

		Matrix M1 = new Matrix(v);

		// Matriz de inversión del eje z en torno al eje x (Norte)
		double w[][] = new double[3][];
		w[0] = new double[] { -1, 0, 0 };
		w[1] = new double[] { 0, 1, 0 };
		w[2] = new double[] { 0, 0, -1 };
		Matrix M2 = new Matrix(w);

		T = M2.times(M1); 
	}

	/**
	 * Actualiza el {@link #centro} y la matriz {@link #T} usando los de la ruta pasada
	 * @param rutaUsar ruta de la que se tomará el centro y T
	 */
	public void actualizaSistemaLocal(Ruta rutaUsar) {
		if(rutaUsar!=null && rutaUsar.T!=null && rutaUsar.centro!=null)
			return;
		T=rutaUsar.T;
		centro=rutaUsar.centro;
	}
	
	/**
	 * @return the ptoCentro
	 */
	public GPSData getPtoCentro() {
		return centro;
	}

	/**
	 * Fija el centro sin actualizar el sistema local.
	 * @param ptoCentro the ptoCentro to set
	 */
	private void setPtoCentro(GPSData ptoCentro) {
		this.centro = ptoCentro;
	}

	/**
	 * @return the t
	 */
	public Matrix getT() {
		return T;
	}

	/**
	 * @param t the t to set
	 */
	private void setT(Matrix t) {
		T = t;
	}
	
	/**
	 * Calcula y actualiza las coordenadas locales del punto pasado.
	 * Hace uso de la {@link #T las matriz de traslación}.
	 * Si no está definido el {@link #centro} o {@link #T} se inicializan a (-1,-1,-1) :-(
	 * @param pto punto a actualizar
	 */
	public void setCoordenadasLocales(GPSData pto) {
		if (T == null || centro==null) {       
			pto.setXLocal(-1);
			pto.setYLocal(-1);
			pto.setZLocal(-1);
			return;
		}

		Matrix res = pto.getCoordECEF().minus(centro.getCoordECEF()); 
		res = T.times(res); //dejamos como vector columna
		pto.setCoordLocal(res);
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
	 * Añade punto pasado a la ruta. Si esta es espcial sólo si esta a {@link #minDistOperativa} de 
	 * el último de la ruta.
	 * Se controla que el número de puntos no supere {@link #tamMaximo}.
	 * @param data punto a añadir.
	 */
	public void add(GPSData data) {
		if(!esEspacial) {
			//para los buffers temporales siempre se añade quitando los primeros si no cabe
			if(puntos.size()>0) 
				puntos.lastElement().calculaAngSpeed(data);
			else { data.setAngulo(0); data.setVelocidad(0); }
			puntos.add(data);
			while(puntos.size()>tamMaximo) puntos.remove(0);
		} else {
			//para los espaciales sólo se considera el punto si está suficientemente lejos de 
			// el último
			if(puntos.size()==0) {
				data.setAngulo(0); data.setVelocidad(0); 
				puntos.add(data);
			} else if (puntos.lastElement().distancia(data)>minDistOperativa) {
				puntos.lastElement().calculaAngSpeed(data);
				puntos.add(data);
				while(puntos.size()>tamMaximo) puntos.remove(0);				
			}
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
		if(i>=puntos.size())
			return null;
		else
			return puntos.elementAt(i);
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {

	}
}
