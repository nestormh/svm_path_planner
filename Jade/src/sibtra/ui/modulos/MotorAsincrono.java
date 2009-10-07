/**
 * 
 */
package sibtra.ui.modulos;

import sibtra.controlcarro.ControlCarro;
import sibtra.gps.GPSData;
import sibtra.gps.Trayectoria;
import sibtra.imu.AngulosIMU;
import sibtra.log.LoggerArrayDoubles;
import sibtra.log.LoggerFactory;
import sibtra.log.LoggerInt;
import sibtra.predictivo.Coche;
import sibtra.ui.VentanasMonitoriza;
import sibtra.ui.defs.CalculoDireccion;
import sibtra.ui.defs.CalculoVelocidad;
import sibtra.ui.defs.DetectaObstaculos;
import sibtra.ui.defs.Motor;
import sibtra.ui.defs.UsuarioTrayectoria;
import sibtra.util.LabelDatoFormato;
import sibtra.util.PanelFlow;
import sibtra.util.SpinnerDouble;
import sibtra.util.ThreadSupendible;
import sibtra.util.UtilCalculos;

/**
 * @author alberto
 *
 */
public class MotorAsincrono implements Motor, UsuarioTrayectoria {
	
	protected String NOMBRE="Motor Asincrono";
	protected String DESCRIPCION="Actualiza modelo del coche cada vez que se recibe un nuevo dato";
	protected VentanasMonitoriza ventanaMonitoriza=null;
	Trayectoria trayActual=null;
	protected CalculoDireccion calculadorDireccion=null;
	protected CalculoVelocidad calculadorVelocidad=null;
	protected DetectaObstaculos[] detectoresObstaculos=null;
	protected PanelAsincrono panel;
	protected Coche modCoche;

	//Parámetros
//	protected int periodoMuestreoMili = 200;
	protected double cotaAngulo=Math.toRadians(30);
	protected double umbralMinimaVelocidad=0.2;
	protected double pendienteFrenado=1.0;
	protected double margenColision=3.0;
	protected double maximoIncrementoVelocidad=0.1;
	
	//Variables 
	protected double consignaVelocidad;
	protected double consignaVolante;
	protected double consignaVelocidadRecibida;
	protected double consignaVolanteRecibida;
	protected double distanciaMinima;
	private ThreadSupendible thCoche;
	private ThreadSupendible thGPS;
	private ThreadSupendible thIMU;
	private long milisUltimo;
	private long milisDelta=0;
	
	//loggers
	protected LoggerArrayDoubles loger;
	protected LoggerInt logerMasCercano;

	public MotorAsincrono() {
		
	}
	
	
	public boolean setVentanaMonitoriza(VentanasMonitoriza ventMonito) {
		if(ventanaMonitoriza!=null) {
			throw new IllegalStateException("Modulo ya inicializado, no se puede volver a inicializar");
		}
		ventanaMonitoriza=ventMonito;
		
		panel=new PanelAsincrono();
		ventanaMonitoriza.añadePanel(panel, getNombre(),false,false);
		
        //inicializamos modelo del coche
        modCoche = new Coche();
		
        //Tenemos que crear un tread por dispositivo de datos
        //Coche
		thCoche=new ThreadSupendible() {
			private int paqActual=0;
			@Override
			protected void accion() {
				ventanaMonitoriza.conexionCarro.esperaNuevosDatos(paqActual);
				//ha llegado paquete al coche
				double vel=ventanaMonitoriza.conexionCarro.getVelocidadMS();
				double vol=ventanaMonitoriza.conexionCarro.getAnguloVolante();
				setVelocidadVolanteCarro(vel,vol);
				//TODO ver si se puede actuar en carro desde otros threads
				actuaEnCarro();
			}
		};
		thCoche.setName(NOMBRE+"Coche");
		//GPS
		thGPS=new ThreadSupendible() {
			private GPSData datoAnterior=null;
			@Override
			protected void accion() {
				//TODO puede ser un dato temporal
				datoAnterior=ventanaMonitoriza.conexionGPS.esperaNuevoDatoEspacial(datoAnterior);
				//ha llegado dato al GPS
				setPosicionCarro(datoAnterior);
			}
		};
		thGPS.setName(NOMBRE+"GPS");
		//IMU
		thIMU=new ThreadSupendible() {
			private AngulosIMU angAnterior=null;
			@Override
			protected void accion() {
				angAnterior=ventanaMonitoriza.conexionIMU.esperaNuevosDatos(angAnterior);
				//ha llegado paquete al coche
				setYawCarro(Math.toRadians(angAnterior.getYaw()) + ventanaMonitoriza.getDesviacionMagnetica());
			}
		};
		thIMU.setName(NOMBRE+"IMU");
		
		//creamos loggers del módulo
		loger=LoggerFactory.nuevoLoggerArrayDoubles(this, "MotorAsincrono");
		loger.setDescripcion("[consignaVolanteRecibida,consignaVolanteAplicada,consignaVelocidadRecibida"
				 +", consignaVelocidadLimitadaRampa, consignaVelocidadAplicada,distanciaMinimaDetectores]");
		logerMasCercano=LoggerFactory.nuevoLoggerInt(this, "IndiceMasCercano");

		return true;
	}

	private synchronized void evolucionaModeloCoche() {
		long milisActual=System.currentTimeMillis();
		milisDelta=milisActual-milisUltimo;
		modCoche.calculaEvolucion((double)milisDelta/1000.0);
		milisUltimo=milisActual;
	}
	
	/** Metodo que se invoca cuando llega información de la IMU. Evoluciona el coche y fija yaw del mismo */
	protected synchronized void setYawCarro(double yawRad) {
		evolucionaModeloCoche();
		modCoche.setYaw(yawRad);
	}

	/** Método que se invoca cada vez que llega nuevo dato del GPS */
	protected synchronized void setPosicionCarro(GPSData datoAnterior) {
		evolucionaModeloCoche();
		modCoche.setX(datoAnterior.getXLocal());
		modCoche.setY(datoAnterior.getYLocal());
	}


	protected synchronized void setVelocidadVolanteCarro(double vel, double vol) {
		evolucionaModeloCoche();
		modCoche.setVelocidad(vel);
		modCoche.setVolante(vol);
	}


	/** activamos el {@link #thCiclico} */
	public void actuar() {
		if(ventanaMonitoriza==null)
			throw new IllegalStateException("Aun no inicializado");
        //Solo podemos actuar si está todo inicializado
        if(calculadorDireccion==null || calculadorVelocidad==null || detectoresObstaculos==null)
        	throw new IllegalStateException("Faltan modulos por inicializar");
        //vemos si hay trayectoria y la apuntamos
        if(ventanaMonitoriza.hayTrayectoria())
        	trayActual=ventanaMonitoriza.getTrayectoriaSeleccionada(this);
		thCoche.activar();
		thGPS.activar();
		thIMU.activar();
	}

	/** suspendemos el {@link #thCiclico} y paramos PID de {@link ControlCarro } */
	public void parar() {
		if(ventanaMonitoriza==null)
			throw new IllegalStateException("Aun no inicializado");
		thCoche.suspender();
		thGPS.suspender();
		thIMU.suspender();
		//paramos el PID de control carro
		ventanaMonitoriza.conexionCarro.stopControlVel();
	}

	/* (sin Javadoc)
	 * @see sibtra.ui.modulos.Modulo#getDescripcion()
	 */
	public String getDescripcion() {
		return DESCRIPCION;
	}

	/* (sin Javadoc)
	 * @see sibtra.ui.modulos.Modulo#getNombre()
	 */
	public String getNombre() {
		return NOMBRE;
	}

	/** Suspendemos el {@link #thCiclico}, quitamos panel, liberamos la trayectoria */
	public void terminar() {
		if(ventanaMonitoriza==null)
			throw new IllegalStateException("Aun no inicializado");
		thCoche.terminar();
		thGPS.terminar();
		thIMU.terminar();
		ventanaMonitoriza.quitaPanel(panel);
		if(trayActual!=null)  //si hemos cogido una trayectoria la liberamos
			ventanaMonitoriza.liberaTrayectoria(this);
		LoggerFactory.borraLogger(loger);
		LoggerFactory.borraLogger(logerMasCercano);
	}

	public void setCalculadorDireccion(CalculoDireccion calDir) {
		calculadorDireccion=calDir;
	}

	public void setCalculadorVelocidad(CalculoVelocidad calVel) {
		calculadorVelocidad=calVel;
	}

	public void setDetectaObstaculos(DetectaObstaculos[] dectObs) {
		detectoresObstaculos=dectObs;
	}
	
	@SuppressWarnings("serial")
	protected class PanelAsincrono extends PanelFlow {
		public PanelAsincrono() {
			super();
//			setLayout(new GridLayout(0,4));
			//TODO Definir los tamaños adecuados o poner layout
			añadeAPanel(new SpinnerDouble(MotorAsincrono.this,"setUmbralMinimaVelocidad",0,6,0.1), "Min Vel");
			añadeAPanel(new SpinnerDouble(MotorAsincrono.this,"setPendienteFrenado",0.1,3,0.1), "Pend Frenado");
			añadeAPanel(new SpinnerDouble(MotorAsincrono.this,"setMargenColision",0.1,10,0.1), "Margen col");
			añadeAPanel(new SpinnerDouble(MotorAsincrono.this,"setMaximoIncrementoVelocidad",0,6,0.1), "Max Inc V");
			añadeAPanel(new SpinnerDouble(MotorAsincrono.this,"setCotaAnguloGrados",5,30,1), "Cota Angulo");
			//TODO ponel labels que muestren la informacion recibida de los otros módulos y la que se aplica.
			añadeAPanel(new LabelDatoFormato(MotorAsincrono.class,"getMilisDelta","%6d ms"), "Delta milis");
			añadeAPanel(new LabelDatoFormato(MotorAsincrono.class,"getConsignaVelocidad","%4.2f m/s"), "Cons Vel");
			añadeAPanel(new LabelDatoFormato(MotorAsincrono.class,"getConsignaVelocidadRecibida","%4.2f m/s"), "Vel Calc");
			añadeAPanel(new LabelDatoFormato(MotorAsincrono.class,"getConsignaVolanteGrados","%4.2f º"), "Cons Vol");
			añadeAPanel(new LabelDatoFormato(MotorAsincrono.class,"getConsignaVolanteRecibidaGrados","%4.2f º"), "Vol Calc");
			añadeAPanel(new LabelDatoFormato(MotorAsincrono.class,"getDistanciaMinima","%4.2f m"), "Dist Min");
			
		}
	}

	protected synchronized void actuaEnCarro() {

		if(trayActual!=null) {
        	//para actulizar en indice del más cercano
        	trayActual.situaCoche(modCoche.getX(), modCoche.getY());
        	logerMasCercano.add(trayActual.indiceMasCercano());
        }
        	

        //Direccion =============================================================
        double consignaVolanteAnterior=consignaVolante;
        consignaVolante=consignaVolanteRecibida=calculadorDireccion.getConsignaDireccion();
        consignaVolante=UtilCalculos.limita(consignaVolante, -cotaAngulo, cotaAngulo);

        double velocidadActual = ventanaMonitoriza.conexionCarro.getVelocidadMS();
        //Cuando está casi parado no tocamos el volante
        if (velocidadActual >= umbralMinimaVelocidad)
        	ventanaMonitoriza.conexionCarro.setAnguloVolante(consignaVolante);

        // Velocidad =============================================================
    	//Guardamos valor para la siguiente iteracion
    	double consignaVelAnterior=consignaVelocidad;
    	
        consignaVelocidad=consignaVelocidadRecibida=calculadorVelocidad.getConsignaVelocidad();
        
        //vemos la minima distancia de los detectores
        distanciaMinima=Double.MAX_VALUE;
        for(int i=0; i<detectoresObstaculos.length; i++)
        	distanciaMinima=Math.min(distanciaMinima, detectoresObstaculos[i].getDistanciaLibre());
        
        double velRampa=(distanciaMinima-margenColision)*pendienteFrenado;
        double consignaVelocidadRampa=consignaVelocidad=Math.min(consignaVelocidad, velRampa);
        
        double incrementoConsigna=consignaVelocidad-consignaVelAnterior;
        if(incrementoConsigna>maximoIncrementoVelocidad)
        	consignaVelocidad=consignaVelAnterior+maximoIncrementoVelocidad;
    	ventanaMonitoriza.conexionCarro.setConsignaAvanceMS(consignaVelocidad);
    	
    	//TODO pueden ser demasiadas actualizaciones (thread aparte o en uno de los lentos ??)
    	panel.actualizaDatos(MotorAsincrono.this);  //actualizamos las etiquetas
    	
    	loger.add(consignaVolanteRecibida,consignaVolante,consignaVelocidadRecibida
    			, consignaVelocidadRampa, consignaVelocidad,distanciaMinima);
		
	}
		
	/** @return modelo del coche que actuliza este motor */
	public Coche getModeloCoche() {
		if(ventanaMonitoriza==null)
			throw new IllegalStateException("Aun no inicializado");
		return modCoche;
	}

	public double getCotaAngulo() {
		return cotaAngulo;
	}

	public void setCotaAngulo(double cotaAngulo) {
		this.cotaAngulo = cotaAngulo;
	}

	public double getCotaAnguloGrados() {
		return Math.toDegrees(cotaAngulo);
	}

	public void setCotaAnguloGrados(double cotaAngulo) {
		this.cotaAngulo = Math.toRadians(cotaAngulo);
	}

	public double getMargenColision() {
		return margenColision;
	}

	public void setMargenColision(double margenColision) {
		this.margenColision = margenColision;
	}

	public double getMaximoIncrementoVelocidad() {
		return maximoIncrementoVelocidad;
	}

	public void setMaximoIncrementoVelocidad(double maximoIncrementoVelocidad) {
		this.maximoIncrementoVelocidad = maximoIncrementoVelocidad;
	}

	public double getPendienteFrenado() {
		return pendienteFrenado;
	}

	public void setPendienteFrenado(double pendienteFrenado) {
		this.pendienteFrenado = pendienteFrenado;
	}

	public double getUmbralMinimaVelocidad() {
		return umbralMinimaVelocidad;
	}

	public void setUmbralMinimaVelocidad(double umbralMinimaVelocidad) {
		this.umbralMinimaVelocidad = umbralMinimaVelocidad;
	}


	/**
	 * @return the consignaVelocidad
	 */
	public double getConsignaVelocidad() {
		return consignaVelocidad;
	}


	/**
	 * @return the consignaVelocidadRecibida
	 */
	public double getConsignaVelocidadRecibida() {
		return consignaVelocidadRecibida;
	}


	/**
	 * @return the consignaVolante
	 */
	public double getConsignaVolanteGrados() {
		return Math.toDegrees(consignaVolante);
	}

	/**
	 * @return the consignaVolante
	 */
	public double getConsignaVolanteRecibidaGrados() {
		return Math.toDegrees(consignaVolanteRecibida);
	}


	/**
	 * @return el distanciaMinima
	 */
	public double getDistanciaMinima() {
		return distanciaMinima;
	}


	/** apuntamos la nueva trayectoria */
	public void nuevaTrayectoria(Trayectoria tra) {
		trayActual=tra;
	}


	/**
	 * @return the milisDelta
	 */
	public long getMilisDelta() {
		return milisDelta;
	}

}
