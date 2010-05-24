/**
 * 
 */
package sibtra.gps;

import java.io.File;
import java.util.Collection;
import java.util.Vector;
import sibtra.util.*;
import sibtra.flota.InterfazFlota;
import sibtra.flota.Prioridades;
import sibtra.imu.DeclinacionMagnetica;
import sibtra.util.UtilCalculos;

/**
 * Carga y gestiona el {@link InterfazFlota} cargando los tramos, localizando el vehículo, etc.
 * 
 * 
 * @author alberto
 *
 */
public class GestionFlota {

	private static final double distanciaATramoAdmisible = 4;

	InterfazFlota interfFlota=null;
	
	Tramos tramos=null;
	
	/** Trayectorias correspondientes a los distintos tramos. Se referencia a {@link #centro} */
	Trayectoria[] trayectorias=null;
	
	/** Punto que se usará como centro para obtener las Trayectorias */
	GPSData centro=null;
	
	/** Vector con los posibles puntos destino */
	Vector<GPSData> destinos=null;

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

	private double declinaM;
	
	public GestionFlota() {
		interfFlota=new InterfazFlota();
	}
	
	/** Carga gel fichero de tramos e inicializa {@link #interfFlota}.
	 * Usando como centro (para las {@link #trayectorias}) el pasado.
	 * @param centro el centro a utilizar. Si es null se usa el de los tramos. 
	 * @return si se pudo cargar e inicializar correctamente
	 */
	public boolean cargaTramos(File fichTramos, GPSData centro) {
		Tramos nuevosTra=null;
		if((nuevosTra=Tramos.cargaTramos(fichTramos))==null) {
			return false; //no se pudieron cargar los tramos correctamente
		}
		tramos=nuevosTra;
		//fijamos el centro
		if(centro!=null)
			setCentro(centro);
		else
			setCentro(tramos.getCentro());
		
		//Cargamos los destinos del fichero asociado (si lo hay)
		if(tramos.getNombFichDestinos()!=null)
			addDestino(new File(tramos.getNombFichDestinos()));
		
		return inicializaTramos(tramos);
	}
	
	/** Carga gel fichero de tramos e inicializa {@link #interfFlota}.
	 * Usando como centro para las {@link #trayectorias} el de las rutas de {@link #tramos}
	 * @return si se pudo cargar e inicializar correctamente
	 */
	public boolean cargaTramos(File fichTramos) {
		return cargaTramos(fichTramos,null);
	}

	/** Preparamos la invocación a inicializacionTramos */
	private boolean inicializaTramos(Tramos tramos) {
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
	
	/** Establece el nuevo {@link #centro} y por lo tanto actualiza las {@link #trayectorias} y {@link #destinos} */
	public void setCentro(GPSData nuevoCentro) {
		if(nuevoCentro==null)
			throw new IllegalArgumentException("El nuevo centro no puede ser null");
		centro=nuevoCentro;
		declinaM=DeclinacionMagnetica.declinacionSegunPosicion(getCentro());
		if(tramos!=null)
			creaTrayectorias();
		//actulizamos los destinos al nuevo centro
		if(destinos!=null)
			for(GPSData pa:destinos)
				pa.calculaLocales(centro);
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
			trayectorias[i]=new Trayectoria(ra,Double.MAX_VALUE,-1,declinaM);
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
		
		//Buscamos los posibles tramos iniciales
		Vector<Integer> posiblesIni=new Vector<Integer>();
		Vector<Double> largoEnIni=new Vector<Double>();
		for(int traIni=0; traIni<tramos.size(); traIni++) {
			Trayectoria ta=trayectorias[traIni];
			ta.situaCoche(posXCoche,posYCoche);
			if(ta.distanciaAlMasCercano()<distanciaATramoAdmisible 
					&& UtilCalculos.diferenciaAngulos(ta.rumbo[ta.indiceMasCercano()],orientacionCoche)<(Math.PI/2) ) {
				//esta dentro de la distacia admisible y no difiere + de 90º en ángulo
				posiblesIni.add(traIni);
				largoEnIni.add(ta.getLargo(0, ta.indiceMasCercano()));
			}
		}
		
		if(posiblesIni.size()==0){
			System.err.println("Posición del coche no está próxima a ningún tramo");
			return null;
		}
		
		//Buscar el tramo y longitud a la que corresponde el destino y aprovechamos para probar todas las posibilidades
		double largoMin=Double.MAX_VALUE;
		String[] nomTramosRutaMin=null;
		for(int indTFin=0; indTFin<tramos.size(); indTFin++) {
			Trayectoria ta=trayectorias[indTFin];
			ta.situaCoche(posXdestino,posYdestino);
			if(ta.distanciaAlMasCercano()<distanciaATramoAdmisible ) {
				//está suficientemente cerca
				double largoEnFin=ta.getLargo(0, ta.indiceMasCercano());
				//probamos desde todos los posibles origenes
				for(int indPosTIni=0; indPosTIni<posiblesIni.size();indPosTIni++) {
					//Pasamos a invocar a InterfazFlota
					String[] nombTramosRuta=interfFlota.calculaRuta(tramos.getNombre(posiblesIni.get(indPosTIni))
							, largoEnIni.get(indPosTIni)
							, tramos.getNombre(indTFin),largoEnFin);
					if(nombTramosRuta==null || nombTramosRuta.length==0)
						continue; //La ruta no es posible
					double largo=interfFlota.dimeUltimaDistanciaCalculada()+largoEnFin
					+(trayectorias[posiblesIni.get(indPosTIni)].getLargo()-largoEnIni.get(indPosTIni));
					System.out.print("Posible ruta de longitud "+largo+":");
					for(String nt:nombTramosRuta) System.out.print(nt+",");
					System.out.println(".");
					System.out.println("Largo calculado a mano:"+largoRuta(nombresAIndices(nombTramosRuta)
							, largoEnIni.get(indPosTIni), largoEnFin));
					if(largo<largoMin) {
						largoMin=largo;
						nomTramosRutaMin=nombTramosRuta;
						largoIni=largoEnIni.get(indPosTIni);
						largoFin=largoEnFin;
					}
				}
			}

		}
		if(nomTramosRutaMin==null){
			System.err.println("No se ha encontrado ruta al destino");
			return null;
		}
		
		System.out.print("La ruta mínima es de longitud "+largoMin+" y formada por ");
		for(String nt:nomTramosRutaMin) System.out.print(nt+",");
		System.out.println(".");
		
		int[] indTramosRuta = nombresAIndices(nomTramosRutaMin);
		return indTramosRuta;
	}

	/**
	 * Calcula los tramos que deben segirse para ir de la posición actual al destino
	 * @param posicion vector de coordenadas locales (x.y)
	 * @param orientacion orientación (yaw) del vehículo en radianes
	 * @param indDest indice del destino elegido de la lista de destinos disponibles.
	 * @return vector con los índices de los tramos que conforman la ruta al destino
	 */
	public int[] indicesTramosADestino(double posXCoche, double posYCoche,
			double orientacionCoche
			, int indDest) {
		if(indDest<0 || indDest>=destinos.size())
			throw new IllegalArgumentException("Indice de destino elegido "+indDest+" fuera del rango posible");
		return indicesTramosADestino(posXCoche, posYCoche, orientacionCoche
				, destinos.get(indDest).getXLocal(), destinos.get(indDest).getYLocal());
	}
	/**
	 * @param nomTramosRutaMin array con los nombres de los tramos
	 * @return array con los índices de los tramos correspondientes
	 */
	private int[] nombresAIndices(String[] nomTramosRutaMin) {
		int[] indTramosRuta=new int[nomTramosRutaMin.length];
		for(int i=0; i<nomTramosRutaMin.length; i++)
			for(int j=0; j<tramos.size(); j++)
				if( nomTramosRutaMin[i].equals(tramos.getNombre(j)) ) {
					indTramosRuta[i]=j;
					break;
				}
		return indTramosRuta;
	}

	/**
	 * Devuelve largo de la ruta formada por esos tramos
	 * @param indicesTramos indices de tramos
	 * @param largoIni largo en tramo inicial
	 * @param largoFin largo en tramo final
	 * @return
	 */
	private double largoRuta(int[] indicesTramos, double largoIni, double largoFin) {
		if(indicesTramos.length==1)
			//estan en el mismo tramo
			return (largoFin-largoIni);
		//largos en los tramos inicial y final
		double largo=largoFin+(trayectorias[indicesTramos[0]].getLargo()-largoIni);
		//largo del resto de tramos
		for(int i=1; i<(indicesTramos.length-1);i++)
			largo+=trayectorias[indicesTramos[i]].getLargo();
		return largo;
	}
	
	/**
	 * Calcula la trayectoria que debe seguirse para ir de la posición actual al destino
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
		if(indicesTramos==null){ //no fue posible calcular los tramos
			System.out.println("No fue posible calcular los tramos");
			return null;
			}
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
		int longitud = 0;
		//calculamos la longitud total de la ruta completa como la suma de los tramos
		for (int i=0;i<indicesTramos.length;i++){
			if(i==indicesTramos.length-1){
				int indFinal = trayectorias[indicesTramos[i]].indiceHastaLargo(largoEnFinal,0);
				longitud = indFinal + longitud;
			}else{
				longitud = trayectorias[indicesTramos[i]].length() + longitud;
			}			
//			System.out.println("la longitud del  tramo "+indicesTramos[i]+" es " + +trayectorias[indicesTramos[i]].length());
		}		
//		System.out.println("Hay " + indicesTramos.length+ " en la trayectoria completa");
//		System.out.println("la longitud total de la ruta completa es " + longitud);
		double[][] puntos = new double[longitud][5]; // x,y,z,velocidad y rumbo
		int cont = 0;
		// Unimos los tramos
		for (int i=0;i<indicesTramos.length;i++){
			if(i==indicesTramos.length-1){
				int indiceFinal = trayectorias[indicesTramos[i]].indiceHastaLargo(largoEnFinal,0);
//				System.out.println("el índice del punto que se encuentra a "+largoEnFinal+" es "+ indiceFinal);
				for(int j=0;j<indiceFinal;j++){
					puntos[cont][0] = trayectorias[indicesTramos[i]].x[j];
					puntos[cont][1] = trayectorias[indicesTramos[i]].y[j];
					puntos[cont][2] = trayectorias[indicesTramos[i]].z[j];
					puntos[cont][3] = trayectorias[indicesTramos[i]].rumbo[j];
					puntos[cont][4] = trayectorias[indicesTramos[i]].velocidad[j];
					
					cont++;				
				}
			}else{
				for(int j=0;j<trayectorias[indicesTramos[i]].length();j++){
					puntos[cont][0] = trayectorias[indicesTramos[i]].x[j];
					puntos[cont][1] = trayectorias[indicesTramos[i]].y[j];
					puntos[cont][2] = trayectorias[indicesTramos[i]].z[j];
					puntos[cont][3] = trayectorias[indicesTramos[i]].rumbo[j];
					puntos[cont][4] = trayectorias[indicesTramos[i]].velocidad[j];			
					cont++;				
				}
			}						
		}		
		Trayectoria trayCompleta = new Trayectoria(puntos);
		trayCompleta.nuevaDistanciaMaxima(0.1);
		trayCompleta = generaRampaFrenado(trayCompleta, 6);
//		for(int k=0;k<trayCompleta.length();k++){
//			System.out.println("vel "+trayCompleta.velocidad[k]);
//		}
		return trayCompleta;
	}
	/**
	 * Método que varía los valores de velocidad de la trayectoria que se le pasa de manera que
	 * el vehículo se detenga en el último punto de la trayectoria. Si la trayectoria es cerrada
	 * se detendrá igualmente en el último punto
	 * @param tr Trayectoria que se desea variar
	 * @param distIniFrenado Distancia a la que se desea que el vehículo comience a frenar
	 * @return Trayectoria modificada
	 */
	public Trayectoria generaRampaFrenado(Trayectoria tr,double distIniFrenado){
		double dist = 0;
		int i = 0;
		for(i=tr.length()-1;dist < distIniFrenado;i--){
			//Encontramos el punto de la trayectoria donde hay que empezar a frenar
			dist = UtilCalculos.distanciaPuntos(tr.x[i],tr.y[i],tr.x[i-1],tr.y[i-1])+dist;
		}
		double velocidadActual = tr.velocidad[i];
		//Calculamos los parámetros de la rampa de frenado
		int numPuntos = tr.length() - i;
		double pendiente = -velocidadActual/distIniFrenado;
		double c = -pendiente*distIniFrenado;
		double T = distIniFrenado/numPuntos;
		double t = T;		
		for(int j=i;j<tr.length();j++){                 
            tr.velocidad[j] = pendiente*t + c;
            t = t + T;
    }
		return tr;
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

	/**
	 * @return the destinos
	 */
	public Vector<GPSData> getDestinos() {
		return destinos;
	}
	
	/** Añade un nuevo punto a la lista de destinos */
	public void addDestino(GPSData nDest) {
		if(destinos==null)
			destinos=new Vector<GPSData>();
		destinos.add(nDest);
	}

	/** Añade todos los puntos del argumento a los destinos */
	public void addDestino(Collection<GPSData> vdest) {
		if(destinos==null)
			destinos=new Vector<GPSData>();
		destinos.addAll(vdest);
	}
	
	/** Añade los destinos cargados del fichero GPX */
	public void addDestino(File fichGPX) {
		addDestino(ManejaGPX.cargaPuntos(fichGPX));
	}
	
}
