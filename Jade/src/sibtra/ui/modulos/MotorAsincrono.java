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
import sibtra.predictivo.Coche;
import sibtra.ui.VentanasMonitoriza;
import sibtra.ui.defs.Motor;
import sibtra.util.LabelDatoFormato;
import sibtra.util.PanelFlow;
import sibtra.util.SpinnerDouble;
import sibtra.util.ThreadSupendible;
import sibtra.util.UtilCalculos;

/**
 * Implementa un motor asíncrono que actualiza el modelo del coche en cuanto le llega la información de alguna de las
 * tres fuentes disponibles:
 * <dl>
 *   <dt>GPS <dd>Fija inmediatamente la posición (x,y) del coche. Debería tener en cueneta la precisión del GPS
 *   <dt>IMU <dd>Fija directamente la orientación del coche
 *   <dt>Coche<dd>Conocida la velocidad y la orientación de las ruedas hace evolucionar el modelo de la bicicleta
 *</dl>
 *
 * @author alberto
 *
 */
public class MotorAsincrono extends MotorTipico implements Motor {
	
	public String getNombre() { return "Motor Asincrono";}
	public String getDescripcion() { return "Actualiza modelo del coche cada vez que se recibe un nuevo dato";}
	protected PanelAsincrono panel;

	//Parámetros
	protected double pendienteFrenado=1.0;
	protected double margenColision=3.0;
	
	//Variables 
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

	public MotorAsincrono() {
		super();
	}
	
	
	public boolean setVentanaMonitoriza(VentanasMonitoriza ventMonito) {
		super.setVentanaMonitoriza(ventMonito);
		
		panel=new PanelAsincrono();
		ventanaMonitoriza.añadePanel(panel, getNombre(),false,false);
		
		
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
		thCoche.setName(getNombre()+"Coche");
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
		thGPS.setName(getNombre()+"GPS");
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
		thIMU.setName(getNombre()+"IMU");
		
		//creamos loggers del módulo
		loger=LoggerFactory.nuevoLoggerArrayDoubles(this, "MotorAsincrono");
		loger.setDescripcion("[consignaVolanteRecibida,consignaVolanteAplicada,consignaVelocidadRecibida"
				 +", consignaVelocidadLimitadaRampa, consignaVelocidadAplicada,distanciaMinimaDetectores]");

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


	/** activamos los threads */
	public void actuar() {
		super.actuar();
		thCoche.activar();
		thGPS.activar();
		thIMU.activar();
	}

	/** suspendemos los threads y paramos PID de {@link ControlCarro } */
	public void parar() {
		super.parar();
		thCoche.suspender();
		thGPS.suspender();
		thIMU.suspender();
		//paramos el PID de control carro
		ventanaMonitoriza.conexionCarro.stopControlVel();
	}

	/** terminamos los threads, quitamos panel, liberamos la trayectoria */
	public void terminar() {
		super.terminar();
		thCoche.terminar();
		thGPS.terminar();
		thIMU.terminar();
		ventanaMonitoriza.quitaPanel(panel);
		LoggerFactory.borraLogger(loger);
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

	public double getPendienteFrenado() {
		return pendienteFrenado;
	}

	public void setPendienteFrenado(double pendienteFrenado) {
		this.pendienteFrenado = pendienteFrenado;
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
