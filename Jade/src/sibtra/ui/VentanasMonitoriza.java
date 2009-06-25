/**
 * 
 */
package sibtra.ui;

import sibtra.controlcarro.ControlCarro;
import sibtra.controlcarro.PanelCarro;
import sibtra.gps.GPSConnectionTriumph;
import sibtra.gps.PanelGPSTriumph;
import sibtra.imu.ConexionSerialIMU;
import sibtra.imu.PanelIMU;
import sibtra.lms.LMSException;
import sibtra.lms.ManejaLMS;
import sibtra.lms.PanelRF;
import sibtra.util.EligeSerial;

/**
 * @author alberto
 *
 */
public class VentanasMonitoriza extends Ventanas {

	private static final int periodoActulizacion = 0;
	
	static final String[] puertosPorDefecto={"/dev/ttyM1" //GPS
		, "/dev/ttyS1" //RF
		,"/dev/ttyUSB0" //IMU
		,"/dev/ttyMI0" //Carro
		};
	
	ControlCarro conexionCarro;
	GPSConnectionTriumph conexionGPS;
	ConexionSerialIMU conexionIMU;  
	ManejaLMS conexionRF;	
	
	PanelGPSTriumph panelGPS;
    PanelIMU panelIMU;
    PanelCarro panelCarro;
    PanelRF panelRF;

    
    /** Abre la conexion a los 4 perifericos y los paneles de monitorizacion
     * @param args nombre de los puertos de: GPS, RF, IMU, Carro 
     */	
	public VentanasMonitoriza(String[] args) {
		super();
        if (args == null || args.length < 4) {
            System.err.println("Son necesarios 4 argumentos con los puertos seriales");
            System.exit(1);
        }

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
//            manLMS.setResolucionAngular((short)100);
            conexionRF.CambiaAModo25();

        } catch (LMSException e) {
            System.err.println("No fue posible conectar o configurar RF");
        }

        //==============================================================================
        // Tenemos todas las conexiones, creamos los paneles básicos
        
        //Panel del GPS
        panelGPS = new PanelGPSTriumph(conexionGPS);
        añadePanel(panelGPS, "GPS",true);
        panelGPS.actulizacionPeridodica(periodoActulizacion);

        //Panel del Coche
        panelCarro=new PanelCarro(conexionCarro);
        añadePanel(panelCarro, "Coche",true);
        panelCarro.actulizacionPeridodica(periodoActulizacion);

        //Panel de la Imu
        panelIMU = new PanelIMU(conexionIMU);
        añadePanel(panelIMU,"IMU",true);
        panelIMU.actulizacionPeridodica(periodoActulizacion);
        
        //Panel del RF
        panelRF=new PanelRF(conexionRF);
        añadePanel(panelRF, "RF", false); //a la izquierda
        panelRF.actualizacionContinua();
        
        //Terminamos la inicialización de Ventanas
        muestraVentanas();
        
	}

	/** Constructor usando los #puertosPorDefecto */
	public VentanasMonitoriza() {
		this(puertosPorDefecto);
	}
	
    /**
     * @param args Seriales para GPS, IMU, RF y Carro. Si no se pasan se usan las por defecto.
     */
    public static void main(String[] args) {   
        if(args==null)
        	new VentanasMonitoriza(); //usara los por defecto
        else {
            String[] puertos=null;
            if ( args.length < 3) {
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
    
}
