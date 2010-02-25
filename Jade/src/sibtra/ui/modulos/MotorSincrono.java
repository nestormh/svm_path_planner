/**
 * 
 */
package sibtra.ui.modulos;

import sibtra.controlcarro.ControlCarro;
import sibtra.log.LoggerArrayDoubles;
import sibtra.log.LoggerFactory;
import sibtra.ui.VentanasMonitoriza;
import sibtra.ui.defs.Motor;
import sibtra.util.LabelDatoFormato;
import sibtra.util.PanelFlow;
import sibtra.util.SpinnerDouble;
import sibtra.util.SpinnerInt;
import sibtra.util.ThreadSupendible;
import sibtra.util.UtilCalculos;

/**
 * @author alberto
 *
 */
public class MotorSincrono extends MotorTipico implements Motor {
	
	public String getNombre() { return "Motor Síncrono"; }
	public String getDescripcion() { return "Ejecuta las acciones de control con un periodo fijo"; }
	protected PanelSincrono panel;
	protected int periodoMuestreoMili=200;
	protected ThreadSupendible thCiclico;

	protected double pendienteFrenado=1.0;
	protected double margenColision=3.0;
	
	protected double consignaVelocidadRecibida;
	protected double consignaVolanteRecibida;
	protected double distanciaMinima;
	//loggers
	protected LoggerArrayDoubles loger;
	
	public MotorSincrono() {super();}
	
	
	public boolean setVentanaMonitoriza(VentanasMonitoriza ventMonito) {
		super.setVentanaMonitoriza(ventMonito);
		
		panel=new PanelSincrono();
		ventanaMonitoriza.añadePanel(panel, getNombre(),false,false);
				
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
		loger=LoggerFactory.nuevoLoggerArrayDoubles(this, "MotorSincrono",100/periodoMuestreoMili);
		loger.setDescripcion("[consignaVolanteRecibida,consignaVolanteAplicada,consignaVelocidadRecibida"
				 +", consignaVelocidadLimitadaRampa, consignaVelocidadAplicada,distanciaMinimaDetectores]");

		return true;
	}

	/** Suspendemos el {@link #thCiclico}, quitamos panel, liberamos la trayectoria */
	public void terminar() {
		super.terminar();
		thCiclico.terminar();
		ventanaMonitoriza.quitaPanel(panel);
		LoggerFactory.borraLogger(loger);
	}

	/** activamos {@link #thCiclico} */
	public void actuar() {
		super.actuar();
		thCiclico.activar();
	}

	/** suspendemos el {@link #thCiclico} y paramos PID de {@link ControlCarro } */
	public void parar() {
		super.parar();
		thCiclico.suspender();
		//paramos el PID de control carro
		ventanaMonitoriza.conexionCarro.stopControlVel();
	}

	protected void accionPeriodica() {
		super.accionPeriodica();
	
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
		
		//Hacemos evolucionar el modelo del coche
	    modCoche.calculaEvolucion(consignaVolante, velocidadActual, (double)periodoMuestreoMili / 1000.0);
	
	
		panel.actualizaDatos(MotorSincrono.this);  //actualizamos las etiquetas
		
		loger.add(consignaVolanteRecibida,consignaVolante,consignaVelocidadRecibida
				, consignaVelocidadRampa, consignaVelocidad,distanciaMinima);
		
	}

	
	@SuppressWarnings("serial")
	protected class PanelSincrono extends PanelFlow {
		public PanelSincrono() {
			super();
//			setLayout(new GridLayout(0,4));
			//TODO Definir los tamaños adecuados o poner layout
			añadeAPanel(new SpinnerDouble(MotorSincrono.this,"setUmbralMinimaVelocidad",0,6,0.1), "Min Vel");
			añadeAPanel(new SpinnerDouble(MotorSincrono.this,"setPendienteFrenado",0.1,3,0.1), "Pend Frenado");
			añadeAPanel(new SpinnerDouble(MotorSincrono.this,"setMargenColision",0.1,10,0.1), "Margen col");
			añadeAPanel(new SpinnerDouble(MotorSincrono.this,"setMaximoIncrementoVelocidad",0,6,0.1), "Max Inc V");
			añadeAPanel(new SpinnerDouble(MotorSincrono.this,"setCotaAnguloGrados",5,30,1), "Cota Angulo");
			añadeAPanel(new SpinnerInt(MotorSincrono.this,"setPeriodoMuestreoMili",20,2000,20), "Per Muest");
			//TODO ponel labels que muestren la informacion recibida de los otros módulos y la que se aplica.
			añadeAPanel(new LabelDatoFormato(MotorSincrono.class,"getConsignaVelocidad","%4.2f m/s"), "Cons Vel");
			añadeAPanel(new LabelDatoFormato(MotorSincrono.class,"getConsignaVelocidadRecibida","%4.2f m/s"), "Vel Calc");
			añadeAPanel(new LabelDatoFormato(MotorSincrono.class,"getConsignaVolanteGrados","%4.2f º"), "Cons Vol");
			añadeAPanel(new LabelDatoFormato(MotorSincrono.class,"getConsignaVolanteRecibidaGrados","%4.2f º"), "Vol Calc");
			añadeAPanel(new LabelDatoFormato(MotorSincrono.class,"getDistanciaMinima","%4.2f m"), "Dist Min");
			
		}
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

	public int getPeriodoMuestreoMili() {
		return periodoMuestreoMili;
	}

	public void setPeriodoMuestreoMili(int periodoMuestreoMili) {
		this.periodoMuestreoMili = periodoMuestreoMili;
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
	public double getConsignaVolanteRecibidaGrados() {
		return Math.toDegrees(consignaVolanteRecibida);
	}


	/**
	 * @return el distanciaMinima
	 */
	public double getDistanciaMinima() {
		return distanciaMinima;
	}

}
