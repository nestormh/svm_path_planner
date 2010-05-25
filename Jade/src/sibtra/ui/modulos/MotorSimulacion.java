package sibtra.ui.modulos;

import sibtra.gps.GPSData;
import sibtra.gps.Trayectoria;
import sibtra.log.LoggerArrayDoubles;
import sibtra.log.LoggerFactory;
import sibtra.predictivo.Coche;
import sibtra.ui.VentanasMonitoriza;
import sibtra.ui.defs.CalculoDireccion;
import sibtra.ui.defs.CalculoVelocidad;
import sibtra.ui.defs.DetectaObstaculos;
import sibtra.ui.defs.ModificadorTrayectoria;
import sibtra.ui.defs.Modulo;
import sibtra.ui.defs.Motor;
import sibtra.ui.defs.SubModuloUsaTrayectoria;
import sibtra.ui.modulos.MotorSincrono.PanelSincrono;
import sibtra.util.LabelDatoFormato;
import sibtra.util.PanelFlow;
import sibtra.util.SpinnerDouble;
import sibtra.util.SpinnerInt;
import sibtra.util.ThreadSupendible;
import sibtra.util.UtilCalculos;

public class MotorSimulacion extends MotorSincrono implements Motor{
	
//	protected PanelSimulacion panel;
//	protected ThreadSupendible thCiclico;
//	protected int periodoMuestreoMili=200;	
//	Coche cocheSim;
//	Coche modCoche;
//	protected LoggerArrayDoubles loger;
	
	public MotorSimulacion(){
		super();
	}
	@Override
	public boolean setVentanaMonitoriza(VentanasMonitoriza ventMonito) {
		super.setVentanaMonitoriza(ventMonito);		
		double posIniXcoche = ventMonito.getTrayectoriaSeleccionada().x[1];
		double posIniYcoche = ventMonito.getTrayectoriaSeleccionada().y[1];
		this.getModeloCoche().setPostura(posIniXcoche,posIniYcoche,0);
//		panel = new PanelSimulacion();
	//	ventanaMonitoriza.añadePanel(panel, getNombre(),false,false);								
		thCiclico=new ThreadSupendible() {
			private long tSig;

			@Override
			protected void accion() {
				//apuntamos cual debe ser el instante siguiente
		        tSig = System.currentTimeMillis() + periodoMuestreoMili;
				accionPeriodica();
		        //esparmos hasta que haya pasado el tiempo convenido
				while (System.currentTimeMillis() < tSig) {
		            try {
		                Thread.sleep(tSig - System.currentTimeMillis());
		            } catch (Exception e) {}
		        }

			}
		};
		thCiclico.setName(getNombre());
		//creamos loggers del módulo
//		loger=LoggerFactory.nuevoLoggerArrayDoubles(this, "MotorSimulacion",100/periodoMuestreoMili);
//		loger.setDescripcion("[consignaVolanteRecibida,consignaVolanteAplicada,consignaVelocidadRecibida"
//				 +", consignaVelocidadLimitadaRampa, consignaVelocidadAplicada,distanciaMinimaDetectores]");

		return true;
	}
	
	/** Implementa lógica del motor usada en cada iteración */
	protected void accionPeriodica() {
		/**  y busca índice del más cercano. */
		trayActual=trayNueva; //por si ha cambiado		
		if(trayActual!=null) {
			//para actulizar en indice del más cercano
			trayActual.situaCoche(modCoche.getX(), modCoche.getY());
			logerMasCercano.add(trayActual.indiceMasCercano());
		}
	
	    //Direccion =============================================================
	    double consignaVolanteAnterior=consignaVolante;
	    consignaVolante=calculadorDireccion.getConsignaDireccion();
	    consignaVolante=UtilCalculos.limita(consignaVolante, -cotaAngulo, cotaAngulo);
	
	    double velocidadActual = modCoche.getVelocidad();
	    //Cuando está casi parado no tocamos el volante
	    if (velocidadActual >= umbralMinimaVelocidad)
//	    	ventanaMonitoriza.conexionCarro.getVelocidadMS();
	    	modCoche.setConsignaVolante(consignaVolante);
	    	
	
	    // Velocidad =============================================================
		//Guardamos valor para la siguiente iteracion
		double consignaVelAnterior=consignaVelocidad;
		
	    consignaVelocidad=calculadorVelocidad.getConsignaVelocidad();
//	    System.out.println("La consigna de velocidad es " + consignaVelocidad);
	    
	    // Por el momento obviamos los detectores
	    
	    //vemos la minima distancia de los detectores
//	    distanciaMinima=Double.MAX_VALUE;
//	    for(int i=0; i<detectoresObstaculos.length; i++)
//	    	distanciaMinima=Math.min(distanciaMinima, detectoresObstaculos[i].getDistanciaLibre());
//	    
//	    double velRampa=(distanciaMinima-margenColision)*pendienteFrenado;
//	    double consignaVelocidadRampa=consignaVelocidad=Math.min(consignaVelocidad, velRampa);
	    
	    double incrementoConsigna=consignaVelocidad-consignaVelAnterior;
	    if(incrementoConsigna>maximoIncrementoVelocidad)
	    	consignaVelocidad=consignaVelAnterior+maximoIncrementoVelocidad;
	    modCoche.setConsignaVelocidad(consignaVelocidad);
//		ventanaMonitoriza.conexionCarro.setConsignaAvanceMS(consignaVelocidad);
		
		//Hacemos evolucionar el modelo del coche
//	    modCoche.calculaEvolucion(consignaVolante, consignaVelocidad, (double)periodoMuestreoMili / 1000.0);
	    modCoche.calculaEvolucion((double)periodoMuestreoMili / 1000.0);
//	    actualizaModeloCoche();
	    super.panel.actualizaDatos(super.panel);
//		panel.actualizaDatos(MotorSimulacion.this);  //actualizamos las etiquetas
		
//		loger.add(consignaVolanteRecibida,consignaVolante,consignaVelocidadRecibida
//				, consignaVelocidadRampa, consignaVelocidad,distanciaMinima);
		
	}
//	@Override
//	protected void actualizaModeloCoche() {
//        	//sacamos los datos del cohe simulado
//        	double x=modCoche.getX();
//        	double y=modCoche.getY();
//        	double angAct = modCoche.getYaw();
//        	double angVolante=modCoche.getVolante();
//        	//TODO Realimentar posición del volante y la velocidad del coche.
//        	modCoche.setPostura(x, y, angAct, angVolante);
////        }
//	}
//	@Override
//	public void actuar() {
//		if(ventanaMonitoriza==null)
//			throw new IllegalStateException("Aun no inicializado");
//	    //Solo podemos actuar si está todo inicializado
//	    if(calculadorDireccion==null || calculadorVelocidad==null || detectoresObstaculos==null)
//	    	throw new IllegalStateException("Faltan modulos por inicializar");
//	    //vemos si hay modulos que necesitan la trayectoria
//	    if(necesitanTrIni.size()>0 || modificadorTr!=null) {
//	    	//obtenos la trayectoria inicial seleccinonada por usuario
//	    	System.out.println("Miramos los módulos que necesitan trayectoria");
//	    	trayInicial=ventanaMonitoriza.getTrayectoriaSeleccionada();
//	    	trayActual=trayNueva=trayInicial;
//	    	for(SubModuloUsaTrayectoria mut:necesitanTrIni)
//	    		//se la comunicamos a los módulos
//	    		mut.setTrayectoriaInicial(trayInicial);
//	    	//y al modificador si lo hay
//		    if(modificadorTr!=null) {
//		    	modificadorTr.setTrayectoriaInicial(trayInicial);
//		    	modificadorTr.actuar(); //avisamos arranque a modificador
//		    }
//	    }
//	}
//
//	@Override
//	public void apuntaNecesitaTrayectoria(SubModuloUsaTrayectoria smutr) {
//		// TODO Auto-generated method stub
//
//	}
//	@Override
//	public Coche getModeloCoche() {
//		if(ventanaMonitoriza==null)
//			throw new IllegalStateException("Aun no inicializado");
//		return modCoche;
//	}

//	@Override
//	public void nuevaTrayectoria(Trayectoria nuevaTr) {
//		// TODO Auto-generated method stub
//
//	}
//	@Override
//	public void parar() {
//		if(ventanaMonitoriza==null)
//			throw new IllegalStateException("Aun no inicializado");
//		if(modificadorTr!=null)
//			modificadorTr.parar();
//	}
//
//	@Override
//	public void setCalculadorDireccion(CalculoDireccion calDir) {
//		// TODO Auto-generated method stub
//		
//	}
//
//	@Override
//	public void setCalculadorVelocidad(CalculoVelocidad calVel) {
//		// TODO Auto-generated method stub
//
//	}
//
//	@Override
//	public void setDetectaObstaculos(DetectaObstaculos[] dectObs) {
//		// TODO Auto-generated method stub
//
//	}
//
//	@Override
//	public void setModificadorTrayectoria(ModificadorTrayectoria modifTr) {
//		// TODO Auto-generated method stub
//
//	}

	@Override
	public String getDescripcion() {
		// TODO Auto-generated method stub
		return "Motor para realizar simulaciones de los distintos algoritmos";
	}

	@Override
	public String getNombre() {
		// TODO Auto-generated method stub
		return "Motor Simulación";
	}

//	@Override
//	public void terminar() {
//		// TODO Auto-generated method stub
//
//	}
	
//	protected class PanelSimulacion extends PanelFlow {
//		public PanelSimulacion() {
//			super();
////			setLayout(new GridLayout(0,4));
//			//TODO Definir los tamaños adecuados o poner layout
//			añadeAPanel(new SpinnerDouble(MotorSimulacion.this,"setUmbralMinimaVelocidad",0,6,0.1), "Min Vel");
//			añadeAPanel(new SpinnerDouble(MotorSimulacion.this,"setPendienteFrenado",0.1,3,0.1), "Pend Frenado");
//			añadeAPanel(new SpinnerDouble(MotorSimulacion.this,"setMargenColision",0.1,10,0.1), "Margen col");
//			añadeAPanel(new SpinnerDouble(MotorSimulacion.this,"setMaximoIncrementoVelocidad",0,6,0.1), "Max Inc V");
//			añadeAPanel(new SpinnerDouble(MotorSimulacion.this,"setCotaAnguloGrados",5,30,1), "Cota Angulo");
//			añadeAPanel(new SpinnerInt(MotorSimulacion.this,"setPeriodoMuestreoMili",20,2000,20), "Per Muest");
////			//TODO ponel labels que muestren la informacion recibida de los otros módulos y la que se aplica.
//			añadeAPanel(new LabelDatoFormato(MotorSimulacion.class,"getConsignaVelocidad","%4.2f m/s"), "Cons Vel");
//			añadeAPanel(new LabelDatoFormato(MotorSimulacion.class,"getConsignaVelocidadRecibida","%4.2f m/s"), "Vel Calc");
//			añadeAPanel(new LabelDatoFormato(MotorSimulacion.class,"getConsignaVolanteGrados","%4.2f º"), "Cons Vol");
//			añadeAPanel(new LabelDatoFormato(MotorSimulacion.class,"getConsignaVolanteRecibidaGrados","%4.2f º"), "Vol Calc");
//			añadeAPanel(new LabelDatoFormato(MotorSimulacion.class,"getDistanciaMinima","%4.2f m"), "Dist Min");
//			
//		}
//	}
//	public double getMargenColision() {
//		return margenColision;
//	}
//
//	public void setMargenColision(double margenColision) {
//		this.margenColision = margenColision;
//	}
//
//	public double getPendienteFrenado() {
//		return pendienteFrenado;
//	}
//
//	public void setPendienteFrenado(double pendienteFrenado) {
//		this.pendienteFrenado = pendienteFrenado;
//	}
//
//	public int getPeriodoMuestreoMili() {
//		return periodoMuestreoMili;
//	}
//
//	public void setPeriodoMuestreoMili(int periodoMuestreoMili) {
//		this.periodoMuestreoMili = periodoMuestreoMili;
//	}
//
//
//	/**
//	 * @return the consignaVelocidadRecibida
//	 */
//	public double getConsignaVelocidadRecibida() {
//		return consignaVelocidadRecibida;
//	}
//
//
//	/**
//	 * @return the consignaVolante
//	 */
//	public double getConsignaVolanteRecibidaGrados() {
//		return Math.toDegrees(consignaVolanteRecibida);
//	}
//
//
//	/**
//	 * @return el distanciaMinima
//	 */
//	public double getDistanciaMinima() {
//		return distanciaMinima;
//	}
}
