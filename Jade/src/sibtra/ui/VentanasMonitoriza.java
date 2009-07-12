/**
 * 
 */
package sibtra.ui;

import java.awt.event.ActionEvent;

import javax.swing.AbstractAction;
import javax.swing.Action;
import javax.swing.BoxLayout;
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
import sibtra.ui.modulos.Motor;
import sibtra.util.EligeSerial;

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

	private PanelEligeModulos panSelModulos;

	private PanelTrayectoria panelTrayectoria;

	private double desviacionMagnetica;

	private Action actGrabarRuta;
	private Action actPararGrabarRuta;

	private JPanel panelGrabar;

	private PanelMuestraRuta panelMuestraRuta;

    
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
			conexionRF.pideBarridoContinuo((short)0,(short) 180, (short)1);
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
        panelGrabar.add(panelMuestraRuta);
        añadePanel(panelGrabar, "Grabar", false, false);
        menuAcciones.add(actGrabarRuta);
        menuAcciones.add(actPararGrabarRuta);
        menuAcciones.addSeparator();

        
        //Añadimos panel de selección de modulos
        panSelModulos=new PanelEligeModulos(this);
        añadePanel(panSelModulos, "Modulos", true);
        
        //Terminamos la inicialización de Ventanas
        muestraVentanas();
        
	}

	/** Constructor usando los #puertosPorDefecto */
	public VentanasMonitoriza() {
		this(puertosPorDefecto);
	}
	
	/** Los calculadores, obstaculos, etc. solicitan la ruta a través de este método. 
	 * Si no hay ninguna seleccionada se tendrá que buscar a un selector de ruta para elegir una
	 * @return la ruta que se va a seguir  
	 */
	public Trayectoria getTrayectoriaSeleccionada() {
		if(panelTrayectoria==null)
			panelTrayectoria=new PanelTrayectoria(this);
		return panelTrayectoria.getTrayectoria();
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

}
