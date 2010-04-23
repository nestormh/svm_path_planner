/**
 * 
 */
package sibtra.gps;

import java.io.File;
import java.util.Vector;

import sibtra.flota.InterfazFlota;
import sibtra.flota.Prioridades;
import sibtra.util.UtilCalculos;

/**
 * Carga y gestiona el {@link InterfazFlota} cargando los tramos, localizando el vehículo, etc.
 * 
 * 
 * @author alberto
 *
 */
public class GestionFlota {

	InterfazFlota interfFlota=null;
	
	Tramos tramos=null;
	
	/** Trayectorias correspondientes a los distintos tramos. Se referencia a {@link #centro} */
	Trayectoria[] trayectorias=null;
	
	/** Punto que se usará como centro para obtener las Trayectorias */
	GPSData centro=null;

	/** Largo dentro del tramo inicial donde se encuentra el origen. 
	 * Se calcula en {@link #indicesTramosADestino(double[], double, double[])}
	 * y se deja como campo para poder ser utilizado 
	 * en {@link #trayectoriaADestino(double[], double, double[])}
	 */
	private double largoIni;

	/** Largo dentro del tramo final donde se encuentra el destino. 
	 * Se calcula en {@link #indicesTramosADestino(double[], double, double[])}
	 * y se deja como campo para poder ser utilizado 
	 * en {@link #trayectoriaADestino(double[], double, double[])}
	 */
	private double largoFin;
	
	public GestionFlota() {
		interfFlota=new InterfazFlota();
	}
	
	/** Carga gel fichero de tramos e inicializa {@link #interfFlota}.
	 * Usando como centro para las {@link #trayectorias} el pasado 
	 * @return si se pudo cargar e inicializar correctamente
	 */
	public boolean cargaTramos(File fichTramos, GPSData centro) {
		Tramos nuevosTra=null;
		if((nuevosTra=Tramos.cargaTramos(fichTramos))==null) {
			return false; //no se pudieron cargar los tramos correctamente
		}
		tramos=nuevosTra;
		
		setCentro(centro); //y se crean trayectorias
		
		return inicializaTramos(tramos);
	}
	
	/** Carga gel fichero de tramos e inicializa {@link #interfFlota}.
	 * Usando como centro para las {@link #trayectorias} el de las rutas de {@link #tramos}
	 * @return si se pudo cargar e inicializar correctamente
	 */
	public boolean cargaTramos(File fichTramos) {
		Tramos nuevosTra=null;
		if((nuevosTra=Tramos.cargaTramos(fichTramos))==null) {
			return false; //no se pudieron cargar los tramos correctamente
		}
		tramos=nuevosTra;
		
		centro=tramos.getRuta(1).getCentro();
		creaTrayectorias();

		return inicializaTramos(tramos);
	}
	
	private boolean inicializaTramos(Tramos tramos) {
		//Preparamos la invocación a inicializacionTramos
		int numTramos=tramos.size();
		String[] nomTramos=new String[numTramos];
		double[] longitudes=new double[numTramos];
		int[][] conexiones=new int[numTramos][numTramos];
		Vector<Prioridades> vectPrioridades=new Vector<Prioridades>();
		Vector<Prioridades> vectOposicion=new Vector<Prioridades>();
		for(int i=0; i<numTramos; i++) {
			nomTramos[i]=tramos.getNombre(i);
			longitudes[i]=trayectorias[i].getLargo();
			for(int j=0; j<numTramos; j++) {
				conexiones[i][j]=tramos.isSiguiente(i, j)?1:0;
				if(tramos.isPrioritatio(i, j))
					vectPrioridades.add(new Prioridades(tramos.getNombre(i),tramos.getNombre(j)));
				if(tramos.isPrioritarioOposicion(i, j))
					vectOposicion.add(new Prioridades(tramos.getNombre(i),tramos.getNombre(j)));
			}				
		}
		//invocamos
		return interfFlota.inicializacionTramos(nomTramos, longitudes, conexiones, vectPrioridades, vectOposicion);
	}
	
	/** Establece el nuevo {@link #centro} y por lo tanto actualiza las {@link #trayectorias} */
	public void setCentro(GPSData centro) {
		if(centro==null)
			throw new IllegalArgumentException("El nuevo centro no puede ser null");
		if(tramos!=null)
			creaTrayectorias();
	}
	
	/** Crea el vector de trayectorias a partir de las rutas de los tramos, 
	 * fijamdo el centro y sin añadir puntos ni mirar si es cerrada 
	 */
	private void creaTrayectorias() {
		trayectorias=new Trayectoria[tramos.size()];
		for(int i=0; i<tramos.size(); i++) {
			Ruta ra=tramos.getRuta(i);
			//tenemos una primera ruta, usamos su centro
			ra.actualizaSistemaLocal(centro);
			ra.actualizaCoordenadasLocales();
			//Obtenemos trayectoria de la ruta sin añadir puntos ni mirar si es cerrada
			trayectorias[i]=new Trayectoria(ra,Double.MAX_VALUE,-1);
		}
	}			
	
	/**
	 * Calcula los tramos que deben segirse para ir de la posición actual al destino
	 * @param posicion vector de coordenadas locales (x.y)
	 * @param orientacion orientación (yaw) del vehículo en radianes
	 * @param destino vector de coordenadas locales (x,y)
	 * @return vector con los índices de los tramos que conforman la ruta al destino
	 */
	public int[] indicesTramosADestino(double posXCoche, double posYCoche,
			double orientacionCoche
			, double posXdestino, double posYdestino) {
		
		//Buscar el tramo y longitud a la que corresponde la posición y orientación actuales
		int traIni=-1;
		largoIni=0; //Posición del punto más cercano dentro del tramo inicial
		double distMin=Double.MAX_VALUE;
		for(int i=0; i<tramos.size(); i++) {
			Trayectoria ta=trayectorias[i];
			ta.situaCoche(posXCoche,posYCoche);
			if(ta.distanciaAlMasCercano()<distMin 
					&& UtilCalculos.diferenciaAngulos(ta.rumbo[ta.indiceMasCercano()],orientacionCoche)<(Math.PI/2) ) {
				//esta dentro de la distancia mínima y no difiere + de 90º en ángulo
				distMin=ta.distanciaAlMasCercano();
				traIni=i;
				largoIni=ta.largo(0, ta.indiceMasCercano());
			}
		}
		
		if(traIni==-1){
			System.err.println("Posición del coche no está próxima a ningún tramo");
			return null;
		}
		
		System.out.println("El tramo de salida es "+tramos.getNombre(traIni)+"("+traIni
				+") en posición " + largoIni + "a distancia "+ distMin);
		
		//Buscar el tramo y longitud a la que corresponde el destino
		int traFin=-1;
		largoFin=0; //Posición del punto más cercano dentro del tramo final
		distMin=Double.MAX_VALUE;
		for(int i=0; i<tramos.size(); i++) {
			Trayectoria ta=trayectorias[i];
			ta.situaCoche(posXdestino,posYdestino);
			if(ta.distanciaAlMasCercano()<distMin ) {
				//esta dentro de la distancia mínima
				distMin=ta.distanciaAlMasCercano();
				traFin=i;
				largoFin=ta.largo(0, ta.indiceMasCercano());
			}
		}
		if(traFin==-1){
			System.err.println("Posición destino no está próxima a ningún tramo");
			return null;
		}
		
		System.out.println("El tramo de destino es "+tramos.getNombre(traFin)+"("+traFin
				+") en posición " + largoFin + "a distancia "+ distMin);
		
		//Pasamos a invocar a InterfazFlota
		String[] nombTramosRuta=interfFlota.calculaRuta(tramos.getNombre(traIni), largoIni
				, tramos.getNombre(traFin),largoFin);
		if(nombTramosRuta==null || nombTramosRuta.length==0)
			return null; //La ruta no es posible
		int[] indTramosRuta=new int[nombTramosRuta.length];
		for(int i=0; i<nombTramosRuta.length; i++)
			for(int j=0; j<tramos.size(); j++)
				if( nombTramosRuta[i].equals(tramos.getNombre(j)) ) {
					indTramosRuta[i]=j;
					break;
					//j=tramos.size(); //para salir del bucle interior
				}
		return indTramosRuta;
	}
	
	/**
	 * Calcula la trayectoria que debe segirse para ir de la posición actual al destino
	 * @param posicion vector de coordenadas locales (x.y)
	 * @param orientacion orientación (yaw) del vehículo en radianes
	 * @param destino vector de coordenadas locales (x,y)
	 * @return Trayectoria a seguir, con rampa de parada en el destino.
	 */
	public Trayectoria trayectoriaADestino(double posXCoche, double posYCoche,
			double orientacionCoche
			, double posXdestino, double posYdestino) {
		int[] indicesTramos=indicesTramosADestino(posXCoche, posYCoche, orientacionCoche
				, posXdestino, posYdestino);
		if(indicesTramos==null) //no fue posible calcular los tramos
			return null;
		return construyeTrayectoriaCompleta(indicesTramos,largoIni,largoFin);
	}
	
	
	/** Contruye trayectoria completa empatando los tramos según el vectro de índices.
	 * termina con rampa de parada en el destino
	 * @param indicesTramos indice de los {@link #tramos} que deben componer la trayectoria
	 * @param largoEnInicial distancia dentro del tramo inicial donde se encuentra el origen
	 * @param largoEnFinal distancia dentro del tramo final donde se encuentra el destino
	 * @return tramo único para llegar de origen a fin
	 * 
	 * @author jesus
	 */
	private Trayectoria construyeTrayectoriaCompleta(int[] indicesTramos, double largoEnInicial, double largoEnFinal) {
		// TODO Auto-generated method stub
		return null;
	}

	/**
	 * @return the centro
	 */
	public GPSData getCentro() {
		return centro;
	}

	/**
	 * @return the interfFlota
	 */
	public InterfazFlota getInterfFlota() {
		return interfFlota;
	}

	/**
	 * @return the tramos
	 */
	public Tramos getTramos() {
		return tramos;
	}

	/**
	 * @return the trayectorias
	 */
	public Trayectoria[] getTrayectorias() {
		return trayectorias;
	}

}
