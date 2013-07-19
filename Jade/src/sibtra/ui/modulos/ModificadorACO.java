package sibtra.ui.modulos;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;

import javax.swing.JButton;

import sibtra.predictivo.Coche;
import sibtra.shm.ShmInterface;

import sibtra.gps.Ruta;
import sibtra.gps.Trayectoria;
import sibtra.ui.VentanasMonitoriza;
//import sibtra.ui.defs.CalculaRuta;
import sibtra.ui.defs.ModificadorTrayectoria;
import sibtra.ui.defs.Motor;
import sibtra.util.LabelDatoFormato;
import sibtra.util.PanelFlow;
import sibtra.util.SpinnerDouble;
import sibtra.util.SpinnerInt;
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
	PanelModACO panelACO;
	private Trayectoria trayectoria;
	String NOMBRE="Modificador ACO";
	String DESCRIPCION="Modifica la trayectoria usando la información de bordes de la carretera";
	private Motor motor;
	private ThreadSupendible thCiclico;
	double distInicio;
	double longitudTramoDesp;
	/** Indice contando a partir del punto más cercano al coche en el que empieza la rampa**/
	int indiceInicial = 10;
	/** Indice contando a partir del punto más cercano al coche en el que acaba la rampa**/
	int indiceFinal = indiceInicial + 20;
	int finalRampa = 0;
	boolean centroDesplazado = false;
	boolean esquivando = false;
	
	Coche modCoche;
	public int umbralDesp = 50;
	public double gananciaLateral = 0.005;
	public double periodoMuestreoMili = 100;
	private Trayectoria trAux;
	private Trayectoria trDesplazada;
	private boolean rampaPasada = false;
	private double ultimoDesp = 0;
	/** Umbral por debajo del cual se considera que el vehículo está bien alineado con la trayectoria**/
	private double umbralRumbo = 5;
	/** Umbral por debajo del cual se considera que el coche está encima de la trayectoria**/
	private double umbralSeparacion = 0.3;
	private double despCentro;
	
	public double getDespCentro() {
		return despCentro;
	}
	public void setDespCentro(double despCentro) {
		this.despCentro = despCentro;
	}
	public double getUmbralRumbo() {
		return umbralRumbo;
	}
	/**
	 * 
	 * @param umbralRumbo Umbral angular en grados
	 */
	public void setUmbralRumbo(double umbralRumbo) {
		this.umbralRumbo = umbralRumbo;
	}
	
	public double getUmbralSeparacion() {
		return umbralSeparacion;
	}
	
	public void setUmbralSeparacion(double umbralSeparacion) {
		this.umbralSeparacion = umbralSeparacion;
	}
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
		trAux = new Trayectoria(trayectoria);
		trDesplazada = new Trayectoria(trayectoria);
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
		panelACO = new PanelModACO();
		ventMonitoriza.añadePanel(panelACO,"Panel ACO",false,false);
		//Le decimos que modelo de coche tiene que usar
		modCoche = motor.getModeloCoche();
		
		thCiclico=new ThreadSupendible() {
			private long tSig;			

			@Override
			protected void accion() {				
				//apuntamos cual debe ser el instante siguiente
		        tSig = System.currentTimeMillis() + (long)periodoMuestreoMili;
		        accionPeriodica();
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
	
	private boolean isAlineado(Trayectoria tr) {
		boolean alineado = false;
		tr.situaCoche(motor.getModeloCoche().getX(),motor.getModeloCoche().getY());
		int masCercano = tr.indiceMasCercano();
		double difRumbo = Math.abs(motor.getModeloCoche().getYaw()-tr.rumbo[masCercano]);
		if((difRumbo < umbralRumbo) && (tr.distanciaAlMasCercano() < umbralSeparacion)){
			alineado = true;
		}else{
			alineado = false;
		}
		return alineado;
		
	}
	
	private void desplazaTrayectoria(double despLateral,int masCercano,
			int indInicial,int indFinal){
		double despX = 0; 
		double despY = 0;
		double dx = trayectoria.x[(indFinal+masCercano)%trayectoria.length()]-
					trayectoria.x[(indInicial+masCercano)%trayectoria.length()];
		double dy = trayectoria.y[(indFinal+masCercano)%trayectoria.length()]-
					trayectoria.y[(indInicial+masCercano)%trayectoria.length()];
		double distanciaRampa = Math.sqrt(dx*dx+dy*dy);
		double despLateralMax = despLateral;
//		double pendiente = Math.abs(despLateralMax)/distanciaRampa;
		double pendiente = despLateralMax/distanciaRampa;
		for(int i=0;i<trayectoria.length();i++){
			despY = -Math.cos(trayectoria.rumbo[i])*despLateral;
			despX = Math.sin(trayectoria.rumbo[i])*despLateral;
			trDesplazada.x[i] = trayectoria.x[i] + despX;
			trDesplazada.y[i] = trayectoria.y[i] + despY;	
			//condición que cumplen los puntos de la trayectoria que se encuentran
			//por delante del coche				
			if(i>(masCercano+indInicial)%trayectoria.length()){
	
				if (i<masCercano+indFinal){ //Sección de la trayectoria en rampa
					double posXrampa = trayectoria.getLargo(indInicial+masCercano, i);
					despLateral = (posXrampa)*pendiente;
				}					
				// Se calcula un desplazamiento lateral perpendicular al rumbo de cada punto 
				despY = -Math.cos(trayectoria.rumbo[i])*despLateral;
				despX = Math.sin(trayectoria.rumbo[i])*despLateral;
				//Se añade el desplazamiento a las coordenadas del punto
				trAux.x[i] = trayectoria.x[i] + despX;
				trAux.y[i] = trayectoria.y[i] + despY;					
				double dxAux = trAux.x[i]-trAux.x[i-1];
				double dyAux = trAux.y[i]-trAux.y[i-1];
				trAux.rumbo[i] = Math.atan2(dyAux,dxAux);
			}else{//no modificamos si los puntos no están por delante del coche
				trAux.x[i] = trayectoria.x[i];
				trAux.y[i] = trayectoria.y[i];
				trAux.rumbo[i] = trayectoria.rumbo[i];
			}								
		}	
	}
	
	private void vuelveTrayectoriaOriginal(double despLateral,int masCercano,
			int indInicial,int indFinal){
		double despX = 0;
		double despY = 0;
//		double dx = trAux.x[(indFinal+masCercano)%trAux.length()]-
//					trAux.x[(indInicial+masCercano)%trAux.length()];
//		double dy = trAux.y[(indFinal+masCercano)%trAux.length()]-
//					trAux.y[(indInicial+masCercano)%trAux.length()];
		double dx = trDesplazada.x[(indFinal+masCercano)%trDesplazada.length()]-
			trDesplazada.x[(indInicial+masCercano)%trDesplazada.length()];
		double dy = trDesplazada.y[(indFinal+masCercano)%trDesplazada.length()]-
			trDesplazada.y[(indInicial+masCercano)%trDesplazada.length()];
		double distanciaRampa = Math.sqrt(dx*dx+dy*dy);
		System.out.println("LA rampa mide " +distanciaRampa);
//		double pendiente = -Math.abs(despLateral)/distanciaRampa;
		double pendiente = -despLateral/distanciaRampa;
		for(int j=0;j<trAux.length();j++){
			
			if(j>=(masCercano)%trAux.length() &&
					   j<(masCercano+indFinal)%trAux.length()){
				if(j>masCercano+indInicial){
//					double posXrampa = trAux.getLargo(indInicial+masCercano,j);
					double posXrampa = trDesplazada.getLargo(indInicial+masCercano,j);
					System.out.println("posicion en la rampa "+posXrampa);
					double despLateralAplicado = (posXrampa)*pendiente;
//					despY = -Math.cos(trAux.rumbo[j])*despLateralAplicado;
//					despX = Math.sin(trAux.rumbo[j])*despLateralAplicado;
					despY = -Math.cos(trDesplazada.rumbo[j])*despLateralAplicado;
					despX = Math.sin(trDesplazada.rumbo[j])*despLateralAplicado;
				}								
				//Se añade el desplazamiento a las coordenadas del punto
				trAux.x[j] = trAux.x[j] + despX;
				trAux.y[j] = trAux.y[j] + despY;
				double dxAux = trAux.x[j]-trAux.x[j-1];
				double dyAux = trAux.y[j]-trAux.y[j-1];
				trAux.rumbo[j] = Math.atan2(dyAux,dxAux);
			}else{
				trAux.x[j] = trayectoria.x[j];
				trAux.y[j] = trayectoria.y[j];
				trAux.rumbo[j] = trayectoria.rumbo[j];
			}
		}
	}
	
	private void accionPeriodica() {
		int distDerecha = ShmInterface.getAcoRightDist();
//		int distDerecha = ShmInterface.getResolucionHoriz()-ShmInterface.getAcoRightDist();
//		System.out.println(ShmInterface.getResolucionHoriz());
		int distIzquierda = ShmInterface.getAcoLeftDist();
		double posCentro = /*(double)distIzquierda +*/ (double)(distDerecha+distIzquierda)/2;
		despCentro = posCentro - (double)ShmInterface.getResolucionHoriz()/2;
//		System.out.println("posición del centro "+centro +"posición izquierda "+distIzquierda
//				+"posición derecha "+ distDerecha);
		double despLateral = 0;
		int masCercano = 0;		
		trayectoria.situaCoche(modCoche.getX(),modCoche.getY());
		masCercano = trayectoria.indiceMasCercano();
		
		if (Math.abs(despCentro) > umbralDesp){
			despLateral = despCentro*gananciaLateral;
			centroDesplazado = true;
		}
		panelACO.actualizaDatos(this);
		
//		if (distIzquierda>umbralDesp){
//			despLateral = distIzquierda*gananciaLateral;    // Cuando el desp es a la izquierda es negativo			
////			finalRampa = masCercano+indiceFinal;
//			centroDesplazado = true;
//		}else if(distDerecha>umbralDesp){
//			despLateral = -distDerecha*gananciaLateral;			
////			finalRampa = masCercano+indiceFinal;
//			centroDesplazado = true;
//		}else{
//			despLateral = 0;			
//			if(masCercano>trayectoria.indiceMasCercano()+indiceFinal){
//			}
//		}
		
		//Comprobamos si el coche ya ha superado la rampa
		if (masCercano > finalRampa){
			rampaPasada  = true;
		}
		
//		System.out.println("Dist Izquierda " + distIzquierda + "\\\\\\ Dist Derecha " + distDerecha);
		//La trayectoria original se le indica al modificadorACO a través del método
		//setTrayectoriaInicial y no se modifica		
		double despX = 0;
		double despY = 0;	
//		setDistInicio(1);
//		setLongitudTramoDesp(4);
//		calculaIndiceInicial();
//		calculaIndiceFinal();
		// Es necesario situar el coche en la ruta antes de buscar el indice más cercano 		
//		int masCercano = trayectoria.indiceMasCercano();
//		if (masCercano>finalRampa){
//			esquivando = false;
//		}
		// Rama del bucle para esquivar
		if((trayectoria.length() != 0) && centroDesplazado && isAlineado(trayectoria)
				&& !esquivando && rampaPasada){
			rampaPasada = false;
			finalRampa = masCercano+indiceFinal;
			esquivando = true;
			centroDesplazado = false;
			System.out.println("Esquivando!!");
			desplazaTrayectoria(despLateral, masCercano,indiceInicial,indiceFinal);
			ultimoDesp  = despLateral;
			motor.nuevaTrayectoria(trAux);
		}
		
		//Rama del bucle para volver a la trayectoria
		if((trayectoria.length() != 0) && centroDesplazado && isAlineado(trAux)
				&& esquivando && rampaPasada){
			System.out.println("Volviendo a la trayectoria original!!");
			rampaPasada = false;
			esquivando = false;
			centroDesplazado = false;
			trAux.situaCoche(modCoche.getX(),modCoche.getY());
			masCercano = trAux.indiceMasCercano();
			finalRampa = masCercano+indiceFinal;
			vuelveTrayectoriaOriginal(ultimoDesp, masCercano,indiceInicial,indiceFinal);
			motor.nuevaTrayectoria(trAux);
		}				
	}

	@Override
	public void terminar() {
		ventanaMonitoriza.quitaPanel(panelACO);
		thCiclico.terminar();		
	}

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
	public double getPeriodoMuestreoMili() {
		return periodoMuestreoMili;
	}

	public void setPeriodoMuestreoMili(double periodoMuestreoMili) {
		this.periodoMuestreoMili = periodoMuestreoMili;
	}
	
	public int getIndiceInicial() {
		return indiceInicial;
	}
	public void setIndiceInicial(int indiceInicial) {
		this.indiceInicial = indiceInicial;
	}
	public int getIndiceFinal() {
		return indiceFinal;
	}
	public void setIndiceFinal(int indiceFinal) {
		this.indiceFinal = indiceFinal;
	}
	
	protected class PanelModACO extends PanelFlow implements ActionListener {
		JButton resetear = new JButton("Resetear");		
		public PanelModACO() {
			super();
			añadeAPanel(resetear,"Reseteo del modificador");
			resetear.addActionListener(this);
//			setLayout(new GridLayout(0,4));
			//TODO Definir los tamaños adecuados o poner layout
			añadeAPanel(new SpinnerDouble(ModificadorACO.this,"setGananciaLateral",0,10,0.001), "Ganancia");
			añadeAPanel(new SpinnerInt(ModificadorACO.this,"setUmbralDesp",0,100,1), "Umbral");
			añadeAPanel(new SpinnerDouble(ModificadorACO.this,"setPeriodoMuestreoMili",0,2000,10), "T Muestreo");
			añadeAPanel(new SpinnerInt(ModificadorACO.this,"setIndiceInicial",0,indiceFinal,1), "Inicio Desp");
			añadeAPanel(new SpinnerInt(ModificadorACO.this,"setIndiceFinal",indiceInicial+10,500,1), "Final Desp");
			añadeAPanel(new SpinnerDouble(ModificadorACO.this,"setUmbralRumbo",0,360,0.1), "Umbral angular");
			añadeAPanel(new SpinnerDouble(ModificadorACO.this,"setUmbralSeparacion",0,100,0.1), "Umbral separación");
			añadeAPanel(new LabelDatoFormato(ModificadorACO.class,"getDespCentro","%4.2f m/s"), "Desp centro");
		}

		@Override
		public void actionPerformed(ActionEvent e) {
			if(e.getSource()==resetear) {
				motor.nuevaTrayectoria(trayectoria);
				centroDesplazado = false;
				rampaPasada = false;
				esquivando = false;
			}
			
		}
	}
}
