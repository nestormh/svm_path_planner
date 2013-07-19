/**
 * 
 */
package sibtra.ui;

import java.awt.Color;
import java.awt.event.ActionEvent;

import javax.swing.AbstractAction;
import javax.swing.Action;
import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSpinner;
import javax.swing.SpinnerNumberModel;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import sibtra.controlcarro.ControlCarro;
import sibtra.controlcarro.PanelCarro;
import sibtra.gps.GPSConnectionTriumph;
import sibtra.gps.PanelGPSTriumph;
import sibtra.gps.PanelGrabarRuta;
import sibtra.gps.PanelMuestraRuta;
import sibtra.gps.Trayectoria;
import sibtra.imu.ConexionSerialIMU;
import sibtra.imu.DeclinacionMagnetica;
import sibtra.imu.PanelIMU;
import sibtra.lms.LMSException;
import sibtra.lms.ManejaLMS221;
import sibtra.lms.PanelRF;
import sibtra.ui.defs.CalculoDireccion;
import sibtra.ui.defs.CalculoVelocidad;
import sibtra.ui.defs.DetectaObstaculos;
import sibtra.ui.defs.ModificadorTrayectoria;
import sibtra.ui.defs.Modulo;
import sibtra.ui.defs.Motor;
import sibtra.ui.defs.SeleccionTrayectoriaInicial;
import sibtra.util.EligeSerial;
import sibtra.util.LabelDatoFormato;
import sibtra.util.ThreadSupendible;

/**
 * Es la aplicación principal del control de Verdino.
 * <br><br>
 * Establece comunicación con el hardware, abre una ventana principal para el monitor grande 
 * y una pequeña para el monitor táctil.
 * En ventana principal se crean paneles para el hardware gestinado:
 * <ul>
 *   <li>GPS: en {@link #conexionGPS} y crea un panel {@link #panelGPS}.
 *   <li>IMU: en {@link #conexionIMU} y crea el panel {@link #panelIMU}
 *   <li>Carro: en {@link #conexionCarro} y crea el panel {@link #panelRF}
 *   <li>RF: en {@link #conexionRF} y crea panel {@link #panelRF}
 * </ul>
 * Otros paneles iniciales son:
 * <ul>
 *   <li>{@link #panelGrabar}: que permite grabar una ruta y salvarla en fichero.
 *   <li>{@link Ventanas#pmLog}: para la gestión de los loggers definidos.
 *   <li> {@link Ventanas#panelResumen}: en el que el usuario puede situar los {@link LabelDatoFormato} que más le interes
 *   <li>{@link #panSelModulos} (del tipo {@link PanelEligeModulos}): Desde donde se gestionan los módulos a utilizar en la ejecución.
 * </ul>
 * Los módulos que es necesario definir para la ejecución son los siguientes:
 * <ul>
 *   <li>Módulo motor: Que debe crear el hilo principal y realiza la actuación sobre el hardware a través de las conexiones
 *   abiertas en {@link VentanasMonitoriza}. ({@link Motor})
 *   <li>Módulo de calculo de la dirección: define la consigna angular inicial para el volente ( {@link CalculoDireccion})
 *   <li>Módulo de cálculo de la velocidad: define la consigna inicial de velocidad del carro ( {@link CalculoVelocidad})
 *   <li>Módulos de detección de obstáculos: devuelven la distacia libre según su sistema de detección. Se pueden seleccionar
 *   varios ( {@link DetectaObstaculos}).
 *   <li>Módulo modificador de trayectoria: pueden introducir modificaciones temporales a la trayectoria inicial que 
 *   está siguiendo el vehículo. Puede no seleccionarse ninguno ( {@link ModificadorTrayectoria}).
 *</ul>
 *
 *Estos módulos, una vez seleccionados, se han de crear y posteriormente activar el funcinamiento del motor.
 *Los módulos que lo deseen pueden añadir paneles a la ventana principal invocando los métodos 
 *{@link Ventanas#añadePanel(JPanel, String) añadePanel}.
 *<br><br>
 *Si alguno de los módulos necesita una trayectoria para su funcinamiento se la pide a su correspondietn motor. 
 *Este, a su vez, se la pide a {@link VentanasMonitoriza}, que abrirá el {@link PanelTrayectoria}. Éste es el encardado de 
 *ofrecer la elección del módulo {@link SeleccionTrayectoriaInicial} que determianará la trayectoria inicial a seguir.
 *
 * <br><br>
 * Cuando está detenida la actuación del motor es posible destruir los módulos creados antes de seleccionar y crear
 * unos distintos. Con la tecla "Refresca Modulos" del {@link PanelEligeModulos} se recargan las clases del 
 * paquete sibtra.ui.modulos con los cual las modificaciones realizadas en las clases hasta ese momento se tendrán en 
 * cuenta. Esto permite la corrección de módulos sin necesidad de salir de la aplicación ;-)
 * <br><br>
 * Se hace la gestión de la Z de emergencia. Se puede conocer su estado a través de {@link #isZPulsada()}. Si es
 * pulsada se detiene la actuación del motor.
 * <br><br>
 * TODO Gesitón de la desviación magnética basada en la localización: ULL, Iter, etc.
 * 
 * @author alberto
 *
 */
public class VentanasMonitoriza extends Ventanas {

	private static final int periodoActulizacion = 200;
	
	static final String[] puertosPorDefecto={"/dev/ttyMI1" //GPS
		,"/dev/ttyUSB0" //IMU
		, "/dev/ttyS1" //RF
		,"/dev/ttyMI0" //Carro
		};
	
	public ControlCarro conexionCarro=null;
	public GPSConnectionTriumph conexionGPS=null;
	public ConexionSerialIMU conexionIMU=null;  
	public ManejaLMS221 conexionRF=null;	
	
	public DeclinacionMagnetica declinaMag=new DeclinacionMagnetica();
	
	PanelGPSTriumph panelGPS;
    PanelIMU panelIMU;
    PanelCarro panelCarro;
    PanelRF panelRF;

	PanelEligeModulos panSelModulos;

	PanelTrayectoria panelTrayectoria;

	private Action actGrabarRuta;
	private Action actPararGrabarRuta;

	private JPanel panelGrabar;

	private PanelMuestraRuta panelMuestraRuta;

	private ThreadSupendible thZeta;

	protected boolean zPulsada=true;

	private JLabel jlZeta;

	private SpinnerNumberModel jspDeclinacionMagnetica;

    
    /** Abre la conexion a los 4 perifericos y los paneles de monitorizacion
     * @param args nombre de los puertos de: GPS, RF, IMU, Carro 
     */	
	public VentanasMonitoriza(String[] args) {
		super();
		if (args != null && args.length >= 4) {

			//Conectamos Carro
			System.out.println("Abrimos conexión al Carro en "+args[3]);
			conexionCarro = new ControlCarro(args[3]);

			if (conexionCarro.isOpen() == false) {
				System.err.println("No se obtuvo Conexion al Carro");            
			}

			//conexión de la IMU
			System.out.println("Abrimos conexión IMU en "+args[1]);
			conexionIMU = new ConexionSerialIMU();
			if (!conexionIMU.ConectaPuerto(args[1], 5)) {
				System.err.println("Problema en conexión serial con la IMU");
				System.exit(1);
			}

			//comunicación con GPS
			System.out.println("Abrimos conexión GPS en "+args[0]);
			try {
				conexionGPS = new GPSConnectionTriumph(args[0]);
				System.out.println("Esperamos por la posición de la base");
				if(conexionGPS.esperaCentroBase(1)) {
					conexionGPS.fijaCentro(conexionGPS.posicionDeLaBase());
					declinaMag.setPosicion(conexionGPS.posicionDeLaBase()); //para elegir la declinacion a aplicar
					System.out.println("Base en "+conexionGPS.posicionDeLaBase());
				} else
					System.err.println("NO se consiguió la posición de la base");
				System.out.println("Comenzamos envío periódico desde GPS");
				conexionGPS.comienzaEnvioPeriodico();
			} catch (Exception e) {
				System.err.println("Problema a crear GPSConnection:" + e.getMessage());
				System.exit(1);
			}
			if (conexionGPS == null) {
				System.err.println("No se obtuvo GPSConnection");
				System.exit(1);
			}
			conexionGPS.setCsIMU(conexionIMU);
			conexionGPS.setCsCARRO(conexionCarro);


			//Conectamos a RF
			System.out.println("Abrimos conexión LMS en "+args[2]);
			try {
				conexionRF = new ManejaLMS221(args[2]);
				conexionRF.setDistanciaMaxima(80);
//				manLMS.setResolucionAngular((short)100);
				conexionRF.CambiaAModo25();

			} catch (LMSException e) {
				System.err.println("No fue posible conectar o configurar RF");
			}
		} else {
			System.err.println("Son necesarios 4 argumentos con los puertos seriales");
//			System.exit(1);
		}

        //==============================================================================
        // Tenemos todas las conexiones, creamos los paneles básicos
        
        //Panel del GPS
        panelGPS = new PanelGPSTriumph(conexionGPS);
        añadePanel(panelGPS, "GPS",false);
        panelGPS.actulizacionPeridodica(periodoActulizacion);

        //Panel del Coche
        panelCarro=new PanelCarro(conexionCarro);
        añadePanel(panelCarro, "Coche",false);
        panelCarro.actulizacionPeridodica(periodoActulizacion);

        //Panel de la Imu
        panelIMU = new PanelIMU(conexionIMU);
		{// spiner para la declinacion magnética
			jspDeclinacionMagnetica=new SpinnerNumberModel(Math.toDegrees(declinaMag.getDeclinacionAplicada()),-90.00,90.00,0.01);
			jspDeclinacionMagnetica.addChangeListener(new ChangeListener() {
				public void stateChanged(ChangeEvent e) {
					//se actualiza la declinacion magnetica pasandola a radianes
					declinaMag.setDeclinacionAplicada(Math.toRadians(jspDeclinacionMagnetica.getNumber().doubleValue()));
				}
			});
			JSpinner jspcv=new JSpinner(jspDeclinacionMagnetica);
			panelIMU.añadeAPanel(jspcv, "Declinacion");
		}

        añadePanel(panelIMU,"IMU",false,false);
        panelIMU.actulizacionPeridodica(periodoActulizacion);
        
        //Panel del RF
        panelRF=new PanelRF(conexionRF);
        añadePanel(panelRF, "RF", true, false); //a la izquierda sin scroll
		try { 		
			conexionRF.pideBarridoContinuo((short)20,(short) 160, (short)1);
		} catch (LMSException e) {
			System.err.println("No fue posible Comenzar con envío continuo");
			System.exit(1);
		}
        panelRF.actualizacionContinua();
        
        //Panel para grabar
        actGrabarRuta=new AccionGrabarRuta();
        actPararGrabarRuta=new AccionPararGrabarRuta();
        panelGrabar=new JPanel();
        panelGrabar.setLayout(new BoxLayout(panelGrabar, BoxLayout.PAGE_AXIS));
        panelGrabar.add(new PanelGrabarRuta(conexionGPS,actGrabarRuta,actPararGrabarRuta));
        panelMuestraRuta=new PanelMuestraRuta(conexionGPS.getBufferEspacial());
        conexionGPS.addGpsEventListener(panelMuestraRuta);
        panelGrabar.add(panelMuestraRuta);
        añadePanel(panelGrabar, "Grabar", false, false);
        menuAcciones.add(actGrabarRuta);
        menuAcciones.add(actPararGrabarRuta);
        
        //Añadimos panel de selección de modulos
        panSelModulos=new PanelEligeModulos(this);
        añadePanel(panSelModulos, "Modulos", true);
        
        //Monitorizacion de la Z
        jlZeta=new JLabel("Zeta");
        jlZeta.setForeground(Color.RED);
        barraMenu.add(Box.createHorizontalGlue());
        barraMenu.add(jlZeta);
        thZeta=new ThreadSupendible() {

			@Override
			protected void accion() {
				if(conexionCarro.getAlarma()==1) {
					if (!zPulsada) {
						//se acaba de pulsar la Z
						//paramos
						panSelModulos.accionParar.actionPerformed(null);
						jlZeta.setForeground(Color.RED);
					}
				} else {
					if(zPulsada) {
						//Se acaba de soltar la Z
						jlZeta.setForeground(Color.GREEN);
					}
				}
				zPulsada=conexionCarro.getAlarma()==1;
	            try {
	                Thread.sleep(10);
	            } catch (Exception e) {}
			}
        };
        thZeta.setName("Periodico VentanaMonitoriza");
        thZeta.activar();
        
        //Terminamos la inicialización de Ventanas
        muestraVentanas();
        
	}

	/** Constructor usando los #puertosPorDefecto */
	public VentanasMonitoriza() {
		this(puertosPorDefecto);
	}
	
	/** Los {@link Motor} (y otros)  solicitan la ruta a través de 
	 * este método.
	 * La primera vez se abre el {@link #panelTrayectoria}.
	 * @return la trayectoria inicial que se va a seguir  
	 */
	public Trayectoria getTrayectoriaSeleccionada() {
		if(panelTrayectoria==null)
			panelTrayectoria=new PanelTrayectoria(this);
		return panelTrayectoria.getTrayectoria();
	}
	
	/** Por si un módulo quiere cambiar la trayectoria actual */
	public void setNuevaTrayectoria(Trayectoria tr) {
		panelTrayectoria.setNuevaTrayectoria(tr);
	}
	
	
	/** @return si hay alguna trayectoria definida (si algún módulo la ha pedido)*/
	public boolean hayTrayectoria() {
		return panelTrayectoria!=null && panelTrayectoria.trayectoriaActual!=null;
	}

	
    /**
     * @param args Seriales para GPS, IMU, RF y Carro. Si no se pasan se usan las por defecto.
     */
    public static void main(String[] args) {   
        if(args==null || args.length==0)
        	new VentanasMonitoriza(); //usara los por defecto
        else {
            String[] puertos=null;
            if ( args.length == 1) {
            	//no se han pasado argumentos suficioentes, pedimos los puertos interactivamente
            	String[] titulos = {"GPS", "IMU", "RF", "Coche"};
            	puertos = new EligeSerial(titulos).getPuertos();
            	if (puertos == null) {
            		System.err.println("No se asignaron los puertos seriales");
            		System.exit(1);
            	}
            } else {
            	puertos = args;
            }
            new VentanasMonitoriza(puertos);
        }
    }
	
//	/** Metodo que deben usar los otros módulos para llegar al módulo motor
//	 * @return el motor seleccionado o null si no hay ninguno.
//	 */
//    public Motor getMotor() {
//    	return panSelModulos.obMotor;
//    }
    
    class AccionGrabarRuta extends AbstractAction {
    	public AccionGrabarRuta(){
    		super("Grabar Ruta");
    		setEnabled(true);
    	}
    
        public void actionPerformed(ActionEvent e) {
        	panelMuestraRuta.setRuta(conexionGPS.getBufferRutaEspacial());
        	actGrabarRuta.setEnabled(false);
        	actPararGrabarRuta.setEnabled(true);
        }
    }
    
    class AccionPararGrabarRuta extends AbstractAction {
    	public AccionPararGrabarRuta() {
			super("Parar Grabar Ruta");
			setEnabled(false);
    	}
        public void actionPerformed(ActionEvent e) {
    		panelMuestraRuta.setRuta(conexionGPS.getBufferEspacial());
        	actGrabarRuta.setEnabled(true);
        	actPararGrabarRuta.setEnabled(false);
        }
    }

	/**
	 * @return el zPulsada
	 */
	public boolean isZPulsada() {
		return zPulsada;
	}

}
