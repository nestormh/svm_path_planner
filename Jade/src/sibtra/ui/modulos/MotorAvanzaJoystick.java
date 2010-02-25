/**
 * 
 */
package sibtra.ui.modulos;

import sibtra.gps.GPSData;
import sibtra.gps.Ruta;
import sibtra.gps.Trayectoria;
import sibtra.predictivo.Coche;
import sibtra.ui.VentanasMonitoriza;
import sibtra.ui.defs.CalculoDireccion;
import sibtra.ui.defs.CalculoVelocidad;
import sibtra.ui.defs.DetectaObstaculos;
import sibtra.ui.defs.ModificadorTrayectoria;
import sibtra.ui.defs.Motor;
import sibtra.ui.defs.SubModuloUsaTrayectoria;
import sibtra.util.LabelDatoFormato;
import sibtra.util.ManejaJoystick;
import sibtra.util.PanelFlow;
import sibtra.util.PanelJoystick;
import sibtra.util.SpinnerDouble;
import sibtra.util.SpinnerInt;
import sibtra.util.ThreadSupendible;
import sibtra.util.UtilCalculos;

/**
 * Motor similar a {@link MotorSincrono} para usar directamente el valor de avanza obtenido del
 * joystick.
 * <strong>No adaptado al nuevo esquema</strong>
 * 
 * @author alberto
 *
 */
public class MotorAvanzaJoystick implements Motor, CalculoVelocidad, CalculoDireccion {
	
	final static String NOMBRE="Avanza Joystick";
	final static String DESCRIPCION="INVALIDO Motor sincrono que usa el avanza obtenido del joystick";
	private VentanasMonitoriza ventanaMonitoriza=null;
	Ruta rutaActual=null;
	private CalculoDireccion calculadorDireccion=null;
	private CalculoVelocidad calculadorVelocidad=null;
	private DetectaObstaculos[] detectoresObstaculos=null;
	private PanelSincrono panel;
	private ThreadSupendible thCiclico;
	private Coche modCoche;
	private long milisActulizacion=200;
	/** Puede ser terminado varias veces, para no repetir el procedimiento */
	private boolean terminado=false;

	//Parámetros
	protected int periodoMuestreoMili = 200;
	protected double cotaAngulo=Math.toRadians(45);
	protected double umbralMinimaVelocidad=0.2;
	protected double pendienteFrenado=1.0;
	protected double margenColision=3.0;
	protected int maximoIncrementoAvance=255;
	
	//Variables 
	protected int avance;
	protected double consignaVolante;
	protected double avanceRecibido;
	protected double consignaVolanteRecibida;
	private ManejaJoystick manJoy;
	private PanelJoystick panJoy;
	private ThreadSupendible thActulizaJoy;
	
	public MotorAvanzaJoystick() {
		
	}
	
	
	public boolean setVentanaMonitoriza(VentanasMonitoriza ventMonito) {
		if(ventanaMonitoriza!=null && ventMonito!=ventanaMonitoriza) {
			throw new IllegalStateException("Modulo ya inicializado, no se puede volver a inicializar en otra ventana");
		}
		if(ventMonito==ventanaMonitoriza)
			//el la misma, no hacemos nada ya que implementa 2 interfaces y puede ser elegido 2 veces
			return true;
		ventanaMonitoriza=ventMonito;
		
		panel=new PanelSincrono();
		ventanaMonitoriza.añadePanel(panel, NOMBRE,false,false);
		//para el joystick
		manJoy=new ManejaJoystick();
		panJoy=new PanelJoystick(manJoy);
		ventanaMonitoriza.añadePanel(panJoy, "Joystick", false,false);
		
		thActulizaJoy=new ThreadSupendible() {

			@Override
			protected void accion() {
				panJoy.actualiza(); //esto ya realiza el pool
				try { Thread.sleep(milisActulizacion); } catch (InterruptedException e) {}
			}
		};
		thActulizaJoy.setName(NOMBRE);
		thActulizaJoy.activar();


        //inicializamos modelo del coche
        modCoche = new Coche();
		
		thCiclico=new ThreadSupendible() {
			private long tSig;

			@Override
			protected void accion() {
				//apuntamos cual debe ser el instante siguiente
	            tSig = System.currentTimeMillis() + periodoMuestreoMili;
	            //Actulizamos el modelo del coche =======================================
	            GPSData pa = ventanaMonitoriza.conexionGPS.getPuntoActualTemporal();
	            if(pa==null) {
	            	System.err.println("Modulo "+NOMBRE+":No tenemos punto GPS con que hacer los cáclulos");
	            	//se usa los valores de la evolución
	            } else {
	            	//sacamos los datos del GPS
	            	double x=pa.getXLocal();
	            	double y=pa.getYLocal();
	            	double angAct = Math.toRadians(pa.getAngulosIMU().getYaw()) + ventanaMonitoriza.getDesviacionMagnetica();
	            	//TODO Realimentar posición del volante y la velocidad del coche.
	            	modCoche.setPostura(x, y, angAct);
	            }

	            //Direccion =============================================================
	            double consignaVolanteAnterior=consignaVolante;
	            consignaVolante=consignaVolanteRecibida=calculadorDireccion.getConsignaDireccion();
	            consignaVolante=UtilCalculos.limita(consignaVolante, -cotaAngulo, cotaAngulo);

	            double velocidadActual = ventanaMonitoriza.conexionCarro.getVelocidadMS();
                //Cuando está casi parado no tocamos el volante
                if (velocidadActual >= umbralMinimaVelocidad)
                	ventanaMonitoriza.conexionCarro.setAnguloVolante(consignaVolante);

                // Avance =============================================================
            	//Guardamos valor para la siguiente iteracion
            	int avanceAnterior=avance;
                avanceRecibido=avance=(int)manJoy.getAvance();
	            double incrementoAvance=avance-avanceAnterior;
	            if(incrementoAvance>maximoIncrementoAvance)
	            	avance=avanceAnterior+maximoIncrementoAvance;
            	ventanaMonitoriza.conexionCarro.Avanza(avance);
            	
            	//Hacemos evolucionar el modelo del coche
                modCoche.calculaEvolucion(consignaVolante, velocidadActual, (double)periodoMuestreoMili / 1000.0);


            	panel.actualizaDatos(MotorAvanzaJoystick.this);  //actualizamos las etiquetas
	            //esparmos hasta que haya pasado el tiempo convenido
				while (System.currentTimeMillis() < tSig) {
	                try {
	                    Thread.sleep(tSig - System.currentTimeMillis());
	                } catch (Exception e) {}
	            }
			}
		};
		thCiclico.setName(NOMBRE);
		return true;
	}

	/** activamos el {@link #thCiclico} */
	public void actuar() {
		if(ventanaMonitoriza==null)
			throw new IllegalStateException("Aun no inicializado");
        //Solo podemos actuar si está todo inicializado
        if(calculadorDireccion==null || calculadorVelocidad==null || detectoresObstaculos==null)
        	throw new IllegalStateException("Faltan modulos por inicializar");
		thCiclico.activar();
	}

	/** suspendemos el {@link #thCiclico} */
	public void parar() {
		if(ventanaMonitoriza==null)
			throw new IllegalStateException("Aun no inicializado");
		thCiclico.suspender();
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

	/** Suspendemos el {@link #thCiclico} y quitamos panel */
	public void terminar() {
		if(ventanaMonitoriza==null)
			throw new IllegalStateException("Aun no inicializado");
		if(terminado) return; //ya fue terminado
		thCiclico.terminar();
		ventanaMonitoriza.quitaPanel(panel);
		ventanaMonitoriza.quitaPanel(panJoy);
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
	
	class PanelSincrono extends PanelFlow {
		public PanelSincrono() {
			super();
//			setLayout(new GridLayout(0,4));
			//TODO Definir los tamaños adecuados o poner layout
			añadeAPanel(new SpinnerDouble(MotorAvanzaJoystick.this,"setUmbralMinimaVelocidad",0,6,0.1), "Min Vel");
			añadeAPanel(new SpinnerDouble(MotorAvanzaJoystick.this,"setPendienteFrenado",0.1,3,0.1), "Pend Frenado");
			añadeAPanel(new SpinnerDouble(MotorAvanzaJoystick.this,"setMargenColision",0.1,10,0.1), "Margen col");
			añadeAPanel(new SpinnerInt(MotorAvanzaJoystick.this,"setMaximoIncrementoAvance",0,255,1), "Max Inc V");
			añadeAPanel(new SpinnerDouble(MotorAvanzaJoystick.this,"setCotaAnguloGrados",5,45,1), "Cota Angulo");
			añadeAPanel(new SpinnerInt(MotorAvanzaJoystick.this,"setPeriodoMuestreoMili",20,2000,20), "Per Muest");
			//TODO ponel labels que muestren la informacion recibida de los otros módulos y la que se aplica.
			añadeAPanel(new LabelDatoFormato(MotorAvanzaJoystick.class,"getAvanceAplicado","%4.2f m/s"), "Cons Vel");
			añadeAPanel(new LabelDatoFormato(MotorAvanzaJoystick.class,"getAvanceRecibido","%4.2f m/s"), "Vel Calc");
			añadeAPanel(new LabelDatoFormato(MotorAvanzaJoystick.class,"getConsignaVolanteGrados","%4.2f º"), "Cons Vol");
			añadeAPanel(new LabelDatoFormato(MotorAvanzaJoystick.class,"getConsignaVolanteRecibidaGrados","%4.2f º"), "Vol Calc");
			
		}
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

	public int getMaximoIncrementoAvance() {
		return maximoIncrementoAvance;
	}

	public void setMaximoIncrementoAvance(int maximoIncrementoAvance) {
		this.maximoIncrementoAvance = maximoIncrementoAvance;
	}

	public double getPendienteFrenado() {
		return pendienteFrenado;
	}

	public void setPendienteFrenado(double pendienteFrenado) {
		this.pendienteFrenado = pendienteFrenado;
	}

	public int getPeriodoMuestreoMili() {
		return periodoMuestreoMili;
	}

	public void setPeriodoMuestreoMili(int periodoMuestreoMili) {
		this.periodoMuestreoMili = periodoMuestreoMili;
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
	public double getAvanceAplicado() {
		return avance;
	}


	/**
	 * @return the consignaVelocidadRecibida
	 */
	public double getAvanceRecibido() {
		return avanceRecibido;
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

	/** @return directamente la velocidad que me da {@link ManejaJoystick} */
	public double getConsignaVelocidad() {
		if(ventanaMonitoriza==null)
			throw new IllegalStateException("Aun no inicializado");
		return manJoy.getVelocidad();
	}

	/** @return directamente el alfa que calcula {@link ManejaJoystick} */
	public double getConsignaDireccion() {
		if(ventanaMonitoriza==null)
			throw new IllegalStateException("Aun no inicializado");
		return manJoy.getAlfa();
	}


	public void apuntaNecesitaTrayectoria(SubModuloUsaTrayectoria smutr) {
		// TODO Auto-generated method stub
		
	}


	public Trayectoria getTrayectoriaActual() {
		// TODO Auto-generated method stub
		return null;
	}


	public void setModificadorTrayectoria(ModificadorTrayectoria modifTr) {
		// TODO Auto-generated method stub
		
	}


	public void setMotor(Motor mtr) {
		// TODO Auto-generated method stub
		
	}

}
