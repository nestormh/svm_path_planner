/**
 * 
 */
package sibtra.ui.modulos;

import java.util.Vector;

import javax.swing.plaf.basic.BasicInternalFrameTitlePane.MaximizeAction;

import sibtra.gps.Ruta;
import sibtra.ui.VentanasMonitoriza;
import sibtra.util.PanelDatos;
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
	private CalculoDireccion calculadorDireccion;
	private CalculoVelocidad calculadorVelocidad;
	private Vector<DetectaObstaculos> detectoresObstaculos;
	private PanelSincrono panel;
	private ThreadSupendible thCiclico;

	//Parámetros
	protected long periodoMuestreoMili = 200;
	protected double cotaAngulo=Math.toRadians(45);
	protected double umbralMinimaVelocidad=0.2;
	protected double pendienteFrenado=1.0;
	protected double margenColision=3.0;
	protected double maximoIncrementoVelocidad=0.1;
	
	//Variables 
	protected double consignaVelAnterior;
	
	public MotorSincrono() {
		
	}
	
	
	public void setVentanaMonitoriza(VentanasMonitoriza ventMonito) {
		ventanaMonitoriza=ventMonito;
		
		panel=new PanelSincrono();
		ventanaMonitoriza.añadePanel(panel, NOMBRE);
		
		thCiclico=new ThreadSupendible() {
			private long tSig;

			@Override
			protected void accion() {
				//apuntamos cual debe ser el instante siguiente
	            tSig = System.currentTimeMillis() + periodoMuestreoMili;
	            //Direccion =============================================================
	            double consignaVolante=calculadorDireccion.getConsignaDireccion();
	            UtilCalculos.limita(consignaVolante, -cotaAngulo, cotaAngulo);

	            double velocidadActual = ventanaMonitoriza.conexionCarro.getVelocidadMS();
                //Cuando está casi parado no tocamos el volante
                if (velocidadActual >= umbralMinimaVelocidad)
                	ventanaMonitoriza.conexionCarro.setAnguloVolante(-consignaVolante);

                // Velocidad =============================================================
	            double consignaVelocidad=calculadorVelocidad.getConsignaVelocidad();
	            
	            //vemos la minima distancia de los detectores
	            double distMinin=Double.MAX_VALUE;
	            for(int i=0; i<detectoresObstaculos.size(); i++)
	            	distMinin=Math.min(distMinin, detectoresObstaculos.elementAt(i).getDistanciaLibre());
	            
	            double velRampa=(distMinin-margenColision)*pendienteFrenado;
	            consignaVelocidad=Math.min(consignaVelocidad, velRampa);
	            
	            double incrementoConsigna=consignaVelocidad-consignaVelAnterior;
	            if(incrementoConsigna>maximoIncrementoVelocidad)
	            	consignaVelocidad=consignaVelAnterior+maximoIncrementoVelocidad;
            	ventanaMonitoriza.conexionCarro.setConsignaAvanceMS(consignaVelocidad);


	            //esparmos hasta que haya pasado el tiempo convenido
				while (System.currentTimeMillis() < tSig) {
	                try {
	                    Thread.sleep(tSig - System.currentTimeMillis());
	                } catch (Exception e) {}
	            }
			}
		};
		thCiclico.setName(NOMBRE);
	}

	/** activamos el {@link #thCiclico} */
	public void actuar() {
		if(ventanaMonitoriza==null)
			throw new IllegalStateException("Aun no inicializado");
		thCiclico.activar();
	}

	/** suspendemos el {@link #thCiclico} */
	public void parar() {
		if(ventanaMonitoriza==null)
			throw new IllegalStateException("Aun no inicializado");
		thCiclico.suspender();
	}

	/* (sin Javadoc)
	 * @see sibtra.ui.modulos.Motor#getRutaSeleccionada()
	 */
	public Ruta getRutaSeleccionada() {
		if(ventanaMonitoriza==null)
			throw new IllegalStateException("Aun no inicializado");
		if(rutaActual!=null) return rutaActual;
		// TODO Buscar los proveedores de ruta, seleccionarlos y elegir una ruta
		return null;
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
		ventanaMonitoriza.quitaPanel(panel);
		thCiclico.suspender();
	}

	public void setCalculadorDireccion(CalculoDireccion calDir) {
		calculadorDireccion=calDir;
	}

	public void setCalculadorVelocidad(CalculoVelocidad calVel) {
		calculadorVelocidad=calVel;
	}

	public void setDetectaObstaculos(Vector<DetectaObstaculos> dectObs) {
		detectoresObstaculos=dectObs;
	}
	
	class PanelSincrono extends PanelDatos {
		public PanelSincrono() {
			añadeAPanel(new SpinnerDouble(MotorSincrono.this,"setUmbralMinimaVelocidad",0,6,0.1), "Min Vel");
			añadeAPanel(new SpinnerDouble(MotorSincrono.this,"setPendienteFrenado",0.1,3,0.1), "Pend Frenado");
			añadeAPanel(new SpinnerDouble(MotorSincrono.this,"setMargenColision",0.1,10,0.1), "Margen col");
			añadeAPanel(new SpinnerDouble(MotorSincrono.this,"setMaximoIncrementoVelodidad",0,6,0.1), "Max Inc V");
			añadeAPanel(new SpinnerDouble(MotorSincrono.this,"setCotaAnguloGrados",5,45,1), "Cota Angulo");
			añadeAPanel(new SpinnerInt(MotorSincrono.this,"setPeriodoMuestreoMili",20,2000,20), "Per Muest");
		}
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

	public long getPeriodoMuestreoMili() {
		return periodoMuestreoMili;
	}

	public void setPeriodoMuestreoMili(long periodoMuestreoMili) {
		this.periodoMuestreoMili = periodoMuestreoMili;
	}

	public double getUmbralMinimaVelocidad() {
		return umbralMinimaVelocidad;
	}

	public void setUmbralMinimaVelocidad(double umbralMinimaVelocidad) {
		this.umbralMinimaVelocidad = umbralMinimaVelocidad;
	}

}
