package sibtra.ui.modulos;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;

import sibtra.predictivo.Coche;
import sibtra.shm.ShmInterface;

import sibtra.gps.Ruta;
import sibtra.gps.Trayectoria;
import sibtra.ui.VentanasMonitoriza;
//import sibtra.ui.defs.CalculaRuta;
import sibtra.ui.defs.ModificadorTrayectoria;
import sibtra.ui.defs.Motor;
import sibtra.util.ThreadSupendible;
/**
 * Clase que implementa {@link ModificadorTrayectoria} y que lee de la memoria compartida los datos 
 * suministrados por el algoritmo ACO y desplaza la trayectoria lateralmente en caso de 
 * presentarse un estrechamiento de la carretera
 * @author jesus
 *
 */
public class ModificadorACO implements ModificadorTrayectoria{
	
	VentanasMonitoriza ventanaMonitoriza;
	private Trayectoria trayectoria;
	String NOMBRE="Modificador ACO";
	String DESCRIPCION="Modifica la trayectoria usando la información de bordes de la carretera";
	private Motor motor;
	private ThreadSupendible thCiclico;
	double distInicio;
	double longitudTramoDesp;
	int indiceInicial;
	int indiceFinal;
	
	Coche modCoche;
	public int umbralDesp = 30;
	public double gananciaLateral = 0.1;
	/**
	 * Seteador de la distancia a partir del coche a la que se desea empezar a desplazar
	 * lateralmente la trayectoria
	 * @param distInicio Distancia en metros
	 */
	public void setDistInicio(double distInicio) {			
		this.distInicio = distInicio;
	}
	/**
	 * Seteador de la longitud deseada para el tramo de la trayectoria que se va a desplazar
	 * @param longitudTramoDesp Longitud en metros
	 */
	public void setLongitudTramoDesp(double longitudTramoDesp) {
		this.longitudTramoDesp = longitudTramoDesp;
	}
	/**
	 * Calcula el índice a partir del cual hay que empezara desplazar lateralmente la 
	 * trayecoria
	 * @return el índice de la trayectoria 
	 */
	public int calculaIndiceInicial(){
		if (trayectoria != null){
			trayectoria.situaCoche(modCoche.getX(),modCoche.getY());
			int indIni = trayectoria.indiceMasCercano();
			this.indiceInicial = trayectoria.indiceHastaLargo(distInicio, indIni);  
			return indiceInicial;
		}else{
			System.out.println("La trayectoria no puede ser null");
			return 0;
		}
	}
	
	/**
	 * Calcula el índice del punto de la trayectoria hasta el cual hay que realizar el desplazamiento
	 * lateral
	 * @return índice de la trayectoria
	 */
	public int calculaIndiceFinal(){
		if (trayectoria != null){
			this.indiceFinal = trayectoria.indiceHastaLargo(longitudTramoDesp, indiceInicial);
		}else {
			System.out.println("La trayectoria no puede ser null");
			return 0;
		}		
		return indiceFinal;
	}
	
	@Override
	public void setTrayectoriaInicial(Trayectoria tra) {
		this.trayectoria = tra;
	}

	@Override
	public void setMotor(Motor mtr) {		
			motor=mtr;		
	}

	public String getDescripcion() {
		return DESCRIPCION;
	}

	public String getNombre() { 
		return NOMBRE;
	}

	@Override
	public boolean setVentanaMonitoriza(VentanasMonitoriza ventMonitoriza) {
		boolean todoBien = false;
		if(ventMonitoriza != null){
			ventanaMonitoriza = ventMonitoriza;
			todoBien = true;
		}
		//Le decimos que modelo de coche tiene que usar
		modCoche = motor.getModeloCoche();
		
		thCiclico=new ThreadSupendible() {
			private long tSig;
			private long periodoMuestreoMili = 1250;

			@Override
			protected void accion() {
				//apuntamos cual debe ser el instante siguiente
		        tSig = System.currentTimeMillis() + periodoMuestreoMili ;
//		        if (calcular){
		        accionPeriodica();
//		        }				
		        //esperamos hasta que haya pasado el tiempo convenido
				while (System.currentTimeMillis() < tSig) {
		            try {
		                Thread.sleep(tSig - System.currentTimeMillis());
		            } catch (Exception e) {}
		        }
			}			
		};
		thCiclico.setName(getNombre());
		return todoBien;
	}
	
	private void accionPeriodica() {
		int distDerecha = ShmInterface.getResolucionHoriz()-ShmInterface.getAcoRightDist();
		System.out.println(ShmInterface.getResolucionHoriz());
		int distIzquierda = ShmInterface.getAcoLeftDist();
		double despLateral = 0;
		if (distIzquierda>umbralDesp){
			despLateral = distIzquierda*gananciaLateral;    // Cuando el desp es a la izquierda es negativo
		}else if(distDerecha>umbralDesp){
			despLateral = -distDerecha*gananciaLateral;
		}else{
			despLateral = 0;
		}
		
		System.out.println("Dist Izquierda " + distIzquierda + "\\\\\\ Dist Derecha " + distDerecha);
		//La trayectoria original se le indica al modificadorACO a través del método
		//setTrayectoriaInicial y no se modifica
		Trayectoria trAux = new Trayectoria(trayectoria); 
		double despX = 0;
		double despY = 0;			
		setDistInicio(1);
		setLongitudTramoDesp(4);
		calculaIndiceInicial();
		calculaIndiceFinal();
		// Es necesario situar el coche en la ruta antes de buscar el indice más cercano 
		trayectoria.situaCoche(modCoche.getX(),modCoche.getY());
		if(trayectoria.length() != 0){
//			for(int i=(trayectoria.indiceMasCercano()+indiceInicial)%trayectoria.length();
//			i<(trayectoria.indiceMasCercano()+indiceFinal)%trayectoria.length();
//			i=(i+1)%trayectoria.length()){			
			//Recorremos todos los puntos de la trayectoria
			for(int i=0;i<trayectoria.length();i++){
				//condición que cumplen los puntos de la trayectoria que se encuentran
				//por delante del coche
				if(i>(trayectoria.indiceMasCercano())%trayectoria.length() &&
				   i<(trayectoria.indiceMasCercano()+40)%trayectoria.length()){
					// Se calcula un desplazamiento lateral perpendicular al rumbo de cada punto 
					despY = -Math.cos(trayectoria.rumbo[i])*despLateral;
					despX = Math.sin(trayectoria.rumbo[i])*despLateral;
					//Se añade el desplazamiento a las coordenadas del punto
					trAux.x[i] = trayectoria.x[i] + despX;
					trAux.y[i] = trayectoria.y[i] + despY;
				}else{//no modificamos si los puntos no están por delante del coche
					trAux.x[i] = trayectoria.x[i];
					trAux.y[i] = trayectoria.y[i];
				}								
			}			
		}
//		Trayectoria trAux = new Trayectoria(trayectoria,0.1);		
		motor.nuevaTrayectoria(trAux);
//		return trayectoria;		
	}
	@Override
	public void terminar() {
		thCiclico.terminar();		
	}
	
//	public static void main(String[] args) {
//		CalculaRutaACO cal =new CalculaRutaACO();
//		String fichero = "Rutas/Universidad/Parq_16_07_cerr";
//		Ruta re;
//    	Trayectoria rutaPruebaRellena;
//    try {
//        File file = new File(fichero);
//        ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file));
//        re = (Ruta) ois.readObject();
//        ois.close();
//        double distMax = 0.5;
//        rutaPruebaRellena = new Trayectoria(re,distMax);
//        System.out.println(rutaPruebaRellena.length());
//        System.out.println("Abrimos el fichero");
//
//    } catch (IOException ioe) {
//        re = new Ruta();
//        rutaPruebaRellena = null;
//        System.err.println("Error al abrir el fichero " + fichero);
//        System.err.println(ioe.getMessage());
//    } catch (ClassNotFoundException cnfe) {
//        re = new Ruta();
//        rutaPruebaRellena = null;
//        System.err.println("Objeto leído inválido: " + cnfe.getMessage());
//    }
//    	cal.setTrayectoriaInicial(rutaPruebaRellena);
//    	while(true){
//    		cal.getTrayectoriaActual();
//    	}	
//	}

	@Override
	public void actuar() {
		thCiclico.activar();		
	}

	@Override
	public void parar() {
		thCiclico.suspender();
		
	}
	public int getUmbralDesp() {
		return umbralDesp;
	}
	public void setUmbralDesp(int umbralDesp) {
		this.umbralDesp = umbralDesp;
	}
	public double getGananciaLateral() {
		return gananciaLateral;
	}
	public void setGananciaLateral(double gananciaLateral) {
		this.gananciaLateral = gananciaLateral;
	}
	
}
