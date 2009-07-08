/**
 * 
 */
package sibtra.ui.modulos;

import java.awt.GridLayout;

import sibtra.gps.GPSData;
import sibtra.gps.Ruta;
import sibtra.predictivo.Coche;
import sibtra.ui.VentanasMonitoriza;
import sibtra.util.LabelDatoFormato;
import sibtra.util.PanelDatos;
import sibtra.util.PanelFlow;
import sibtra.util.SpinnerDouble;
import sibtra.util.SpinnerInt;
import sibtra.util.ThreadSupendible;
import sibtra.util.UtilCalculos;

/**
 * @author alberto
 *
 */
public class MotorSincrono implements Motor {
	
	final static String NOMBRE="Motor Sincrono";
	final static String DESCRIPCION="Ejecuta las acciones de control con un periodo fijo";
	private VentanasMonitoriza ventanaMonitoriza=null;
	Ruta rutaActual=null;
	private CalculoDireccion calculadorDireccion=null;
	private CalculoVelocidad calculadorVelocidad=null;
	private DetectaObstaculos[] detectoresObstaculos=null;
	private PanelSincrono panel;
	private ThreadSupendible thCiclico;
	private Coche modCoche;

	//Parámetros
	protected int periodoMuestreoMili = 200;
	protected double cotaAngulo=Math.toRadians(45);
	protected double umbralMinimaVelocidad=0.2;
	protected double pendienteFrenado=1.0;
	protected double margenColision=3.0;
	protected double maximoIncrementoVelocidad=0.1;
	
	//Variables 
	protected double consignaVelocidad;
	protected double consignaVolante;
	protected double consignaVelocidadRecibida;
	protected double consignaVolanteRecibida;
	
	public MotorSincrono() {
		
	}
	
	
	public boolean setVentanaMonitoriza(VentanasMonitoriza ventMonito) {
		if(ventanaMonitoriza!=null) {
			throw new IllegalStateException("Modulo ya inicializado, no se puede volver a inicializar");
		}
		ventanaMonitoriza=ventMonito;
		
		panel=new PanelSincrono();
		ventanaMonitoriza.añadePanel(panel, NOMBRE,false,false);
		
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
                	ventanaMonitoriza.conexionCarro.setAnguloVolante(-consignaVolante);

                // Velocidad =============================================================
            	//Guardamos valor para la siguiente iteracion
            	double consignaVelAnterior=consignaVelocidad;
            	
	            consignaVelocidad=consignaVelocidadRecibida=calculadorVelocidad.getConsignaVelocidad();
	            
	            //vemos la minima distancia de los detectores
	            double distMinin=Double.MAX_VALUE;
	            for(int i=0; i<detectoresObstaculos.length; i++)
	            	distMinin=Math.min(distMinin, detectoresObstaculos[i].getDistanciaLibre());
	            
	            double velRampa=(distMinin-margenColision)*pendienteFrenado;
	            consignaVelocidad=Math.min(consignaVelocidad, velRampa);
	            
	            double incrementoConsigna=consignaVelocidad-consignaVelAnterior;
	            if(incrementoConsigna>maximoIncrementoVelocidad)
	            	consignaVelocidad=consignaVelAnterior+maximoIncrementoVelocidad;
            	ventanaMonitoriza.conexionCarro.setConsignaAvanceMS(consignaVelocidad);
            	
            	//Hacemos evolucionar el modelo del coche
                modCoche.calculaEvolucion(consignaVolante, velocidadActual, (double)periodoMuestreoMili / 1000.0);


            	panel.actualizaDatos(MotorSincrono.this);  //actualizamos las etiquetas
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
		thCiclico.terminar();
		ventanaMonitoriza.quitaPanel(panel);
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
			añadeAPanel(new SpinnerDouble(MotorSincrono.this,"setUmbralMinimaVelocidad",0,6,0.1), "Min Vel");
			añadeAPanel(new SpinnerDouble(MotorSincrono.this,"setPendienteFrenado",0.1,3,0.1), "Pend Frenado");
			añadeAPanel(new SpinnerDouble(MotorSincrono.this,"setMargenColision",0.1,10,0.1), "Margen col");
			añadeAPanel(new SpinnerDouble(MotorSincrono.this,"setMaximoIncrementoVelocidad",0,6,0.1), "Max Inc V");
			añadeAPanel(new SpinnerDouble(MotorSincrono.this,"setCotaAnguloGrados",5,45,1), "Cota Angulo");
			añadeAPanel(new SpinnerInt(MotorSincrono.this,"setPeriodoMuestreoMili",20,2000,20), "Per Muest");
			//TODO ponel labels que muestren la informacion recibida de los otros módulos y la que se aplica.
			añadeAPanel(new LabelDatoFormato(MotorSincrono.class,"getConsignaVelocidad","%4.2f m/s"), "Cons Vel");
			añadeAPanel(new LabelDatoFormato(MotorSincrono.class,"getConsignaVelocidadRecibida","%4.2f m/s"), "Vel Calc");
			añadeAPanel(new LabelDatoFormato(MotorSincrono.class,"getConsignaVolanteGrados","%4.2f º"), "Cons Vol");
			añadeAPanel(new LabelDatoFormato(MotorSincrono.class,"getConsignaVolanteRecibidaGrados","%4.2f º"), "Vol Calc");
			
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

}
