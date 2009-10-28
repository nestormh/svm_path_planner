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

import sibtra.controlcarro.ControlCarro;
import sibtra.controlcarro.PanelCarro;
import sibtra.gps.GPSConnectionTriumph;
import sibtra.gps.PanelGPSTriumph;
import sibtra.gps.PanelGrabarRuta;
import sibtra.gps.PanelMuestraRuta;
import sibtra.gps.Trayectoria;
import sibtra.imu.ConexionSerialIMU;
import sibtra.imu.PanelIMU;
import sibtra.lms.LMSException;
import sibtra.lms.ManejaLMS;
import sibtra.lms.PanelRF;
import sibtra.ui.defs.Motor;
import sibtra.ui.defs.UsuarioTrayectoria;
import sibtra.util.EligeSerial;
import sibtra.util.ThreadSupendible;

/**
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
	public ManejaLMS conexionRF=null;	
	
	PanelGPSTriumph panelGPS;
    PanelIMU panelIMU;
    PanelCarro panelCarro;
    PanelRF panelRF;

	PanelEligeModulos panSelModulos;

	PanelTrayectoria panelTrayectoria;

	private double desviacionMagnetica;

	private Action actGrabarRuta;
	private Action actPararGrabarRuta;

	private JPanel panelGrabar;

	private PanelMuestraRuta panelMuestraRuta;

	private ThreadSupendible thZeta;

	protected boolean zPulsada=true;

	private JLabel jlZeta;

    
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
				if(conexionGPS.esperaCentroBase(20)) {
					conexionGPS.fijaCentro(conexionGPS.posicionDeLaBase());
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
				conexionRF = new ManejaLMS(args[2]);
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
	
	/** Los {@link UsuarioTrayectoria} (calculadores, motores, etc.) solicitan la ruta a través de 
	 * este método.
	 * La primera vez se abre el {@link #panelTrayectoria}.
	 * @param objUsaTr necesario conocer el objeto que solicita la ruta para avisarle cuando hay cambio 
	 * @return la ruta que se va a seguir  
	 */
	public Trayectoria getTrayectoriaSeleccionada(UsuarioTrayectoria objUsaTr) {
		if(panelTrayectoria==null)
			panelTrayectoria=new PanelTrayectoria(this);
		return panelTrayectoria.getTrayectoria(objUsaTr);
	}
	
	/** Por si un módulo quiere cambiar la trayectoria actual */
	public void setNuevaTrayectoria(Trayectoria tr) {
		panelTrayectoria.setNuevaTrayectoria(tr);
	}
	
	/** Indica que no va ha necesitar más la trayectoria. 
	 * Se invocará cuando un módulo vaya a destruirse. 
	 */
	public void liberaTrayectoria(UsuarioTrayectoria objUsaTr) {
		if(panelTrayectoria==null)
			throw new IllegalStateException("Se trata de liberar trayectoria no solicitada");
		panelTrayectoria.liberaTrayectoria(objUsaTr);
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

	public void setDesviacionMagnetica(double desMag) {
		// TODO para recibir desviacion magnetica recibida de fichero
		desviacionMagnetica=desMag;
	}

	/**
	 * @return the desviacionMagnetica
	 */
	public double getDesviacionMagnetica() {
		return desviacionMagnetica;
	}
	
	/** Metodo que deben usar los otros módulos para llegar al módulo motor
	 * @return el motor seleccionado o null si no hay ninguno.
	 */
    public Motor getMotor() {
    	return panSelModulos.obMotor;
    }
    
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
