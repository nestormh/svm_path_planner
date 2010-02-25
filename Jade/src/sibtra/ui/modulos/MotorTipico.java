package sibtra.ui.modulos;

import java.util.Vector;

import sibtra.gps.GPSData;
import sibtra.gps.Trayectoria;
import sibtra.log.LoggerFactory;
import sibtra.log.LoggerInt;
import sibtra.predictivo.Coche;
import sibtra.ui.VentanasMonitoriza;
import sibtra.ui.defs.CalculoDireccion;
import sibtra.ui.defs.CalculoVelocidad;
import sibtra.ui.defs.DetectaObstaculos;
import sibtra.ui.defs.ModificadorTrayectoria;
import sibtra.ui.defs.SubModuloUsaTrayectoria;

public abstract class MotorTipico {

	protected VentanasMonitoriza ventanaMonitoriza = null;

	/** Apunta los módulos que necesitan trayectoria, y se han apuntado 
	 * con {@link #apuntaNecesitaTrayectoria(SubModuloUsaTrayectoria)}
	 * Cada vez que comienze {@link #actuar()}, se les pasará la trayectoria inicial 
	 * invocando {@link SubModuloUsaTrayectoria#setTrayectoriaInicial(Trayectoria)}
	 * */
	protected Vector<SubModuloUsaTrayectoria> necesitanTrIni=new Vector<SubModuloUsaTrayectoria>();
	
	/** La que obtenemos de {@link VentanasMonitoriza} al comenzar {@link #actuar()} */
	protected Trayectoria trayInicial = null;
	/** La úlima que nos ha devuelto el {@link #modificadorTr}, si lo tenemos,
	 * Caso contrario coincidirá con {@link #trayActual} */
	protected Trayectoria trayActual = null;
	protected CalculoDireccion calculadorDireccion = null;
	protected CalculoVelocidad calculadorVelocidad = null;
	protected DetectaObstaculos[] detectoresObstaculos = null;
	protected ModificadorTrayectoria modificadorTr = null;
	protected Coche modCoche;

	//Parámetros
	protected double umbralMinimaVelocidad=0.2;
	protected double cotaAngulo=Math.toRadians(30);
	protected double maximoIncrementoVelocidad=0.1;

	protected double consignaVelocidad;
	protected double consignaVolante;

	/** Para apuntar el índice del punto más cercano de la trayectoria de
	 * donde se encuentra el coche
	 */
	protected LoggerInt logerMasCercano;

	
	public MotorTipico() {
		super();
	}

	/** apuntamos ventana monitoriza e 
	 * inicializamos {@link #modCoche} y {@link #logerMasCercano} 
	 */
	public boolean setVentanaMonitoriza(VentanasMonitoriza ventMonito) {
		if(ventanaMonitoriza!=null) {
			throw new IllegalStateException("Modulo ya inicializado, no se puede volver a inicializar");
		}
		ventanaMonitoriza=ventMonito;
		
        //inicializamos modelo del coche
        modCoche = new Coche();	
        
        //suponemos una 5 muestras segundo
		logerMasCercano=LoggerFactory.nuevoLoggerInt(this, "IndiceMasCercano", 5);

        return true;
	}
	
	/** Comprobamos que esté todo y obtenemos trayectoria inicial */
	public void actuar() {
		if(ventanaMonitoriza==null)
			throw new IllegalStateException("Aun no inicializado");
	    //Solo podemos actuar si está todo inicializado
	    if(calculadorDireccion==null || calculadorVelocidad==null || detectoresObstaculos==null)
	    	throw new IllegalStateException("Faltan modulos por inicializar");
	    //vemos si hay modulos que necesitan la trayectoria
	    if(necesitanTrIni.size()>0 || modificadorTr!=null) {
	    	//obtenos la trayectoria inicial seleccinonada por usuario
	    	trayInicial=ventanaMonitoriza.getTrayectoriaSeleccionada();
	    	trayActual=trayInicial;
	    	for(SubModuloUsaTrayectoria mut:necesitanTrIni)
	    		//se la comunicamos a los módulos
	    		mut.setTrayectoriaInicial(trayInicial);
	    	//y al modificador si lo hay
		    if(modificadorTr!=null)
		    	modificadorTr.setTrayectoriaInicial(trayInicial);
	    }
	}

	/** comprobamos que estemos inicializados */
	public void parar() {
		if(ventanaMonitoriza==null)
			throw new IllegalStateException("Aun no inicializado");
	}

	/** comprobamos que estemos inicializados */
	public void terminar() {
		if(ventanaMonitoriza==null)
			throw new IllegalStateException("Aun no inicializado");
		LoggerFactory.borraLogger(logerMasCercano);
	}

	public String getDescripcion() {
		return "Base para un motor típico";
	}

	public String getNombre() {
		return  "Motor Tipico";
	}

	/** Los {@link SubModuloUsaTrayectoria} se apuntan en el motor para recibir la trayectoria inicial al 
	 * comenzar a actuar. Con ello el motor sabrá 
	 * que hay alguno que necesita trayectorias y se las pedirá a {@link VentanasMonitoriza}
	 * <br>
	 * Necesitamos apunta el módulo en vector {@link #necesitanTrIni}
	 */
	public void apuntaNecesitaTrayectoria(SubModuloUsaTrayectoria smutr) {
		necesitanTrIni.add(smutr);
		if(trayInicial==null) {
	    	//obtenos la trayectoria inicial seleccinonada por usuario
			//Lo hacemos aquí para que no esperar a la primera acción
	    	trayInicial=ventanaMonitoriza.getTrayectoriaSeleccionada();
		}
	}

	/** Devolvemos la {@link #trayActual}.
	 * Si tenemos {@link ModificadorTrayectoria} se la hemos pedimos a él 
	 * al principio de la iteración, caso contrario, conincidirá con la inicial
	 */
	public Trayectoria getTrayectoriaActual() {
		return trayActual;
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
	
	public void setModificadorTrayectoria(ModificadorTrayectoria modifTr) {
		modificadorTr=modifTr;
		if(trayInicial==null) {
	    	//obtenos la trayectoria inicial seleccinonada por usuario
			//Lo hacemos aquí para que no esperar a la primera acción
	    	trayInicial=ventanaMonitoriza.getTrayectoriaSeleccionada();
		}
	}

	protected void actualizaModeloCoche() {
        //Actulizamos el modelo del coche =======================================
        GPSData pa = ventanaMonitoriza.conexionGPS.getPuntoActualTemporal();
        if(pa==null) {
        	System.err.println("Modulo "+getNombre()+":No tenemos punto GPS con que hacer los cáclulos");
        	//se usa los valores de la evolución
        } else {
        	//sacamos los datos del GPS
        	double x=pa.getXLocal();
        	double y=pa.getYLocal();
        	double angAct = Math.toRadians(pa.getAngulosIMU().getYaw()) + ventanaMonitoriza.getDesviacionMagnetica();
        	double angVolante=ventanaMonitoriza.conexionCarro.getAnguloVolante();
        	//TODO Realimentar posición del volante y la velocidad del coche.
        	modCoche.setPostura(x, y, angAct, angVolante);
        }

	}

	/** Actuliza el modelo del coche, pide trayectoria modificada y busca índice del más cercano. */
	protected void accionPeriodica() {
		actualizaModeloCoche();
		if(modificadorTr!=null) {
			Trayectoria nuevaTr=modificadorTr.getTrayectoriaActual();
			if(nuevaTr!=null)  {
				trayActual=nuevaTr;
				for(SubModuloUsaTrayectoria sut: necesitanTrIni)
					sut.setTrayectoriaModificada(trayActual);
			}
		}
	    if(trayActual!=null) {
	    	//para actulizar en indice del más cercano
	    	trayActual.situaCoche(modCoche.getX(), modCoche.getY());
	    	logerMasCercano.add(trayActual.indiceMasCercano());
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

	/**
	 * @return the consignaVelocidad
	 */
	public double getConsignaVelocidad() {
		return consignaVelocidad;
	}

	/**
	 * @return the consignaVolante
	 */
	public double getConsignaVolanteGrados() {
		return Math.toDegrees(consignaVolante);
	}

	public double getUmbralMinimaVelocidad() {
		return umbralMinimaVelocidad;
	}

	public void setUmbralMinimaVelocidad(double umbralMinimaVelocidad) {
		this.umbralMinimaVelocidad = umbralMinimaVelocidad;
	}

	public double getMaximoIncrementoVelocidad() {
		return maximoIncrementoVelocidad;
	}

	public void setMaximoIncrementoVelocidad(double maximoIncrementoVelocidad) {
		this.maximoIncrementoVelocidad = maximoIncrementoVelocidad;
	}

}