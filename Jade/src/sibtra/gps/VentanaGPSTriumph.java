package sibtra.gps;

import java.awt.BorderLayout;
import javax.swing.JFrame;

import sibtra.log.VentanaLoggers;
import sibtra.util.EligeSerial;

/**
 * Ventana con información del GPS Triumph mediante un {@link PanelGPSTriumph}.
 * Activa el auto-refresco del panel, NO utiliza los eventos del GPS.
 * @author alberto
 *
 */
public class VentanaGPSTriumph extends JFrame {

	private PanelGPSTriumph pGPST;
	
	private GPSConnectionTriumph gpsCT;

	/** Milisegundos del periodo de actualización */
	private int milisPeriodo=550;


	public VentanaGPSTriumph(GPSConnectionTriumph gct) {
		super("VentanaGPSTriumph");
		if(gct==null)
			throw new IllegalArgumentException("Conexion a GPS no puede ser null");
		gpsCT=gct;
		pGPST=new PanelGPSTriumph(gpsCT);
		add(pGPST,BorderLayout.CENTER);
		pack();
		setVisible(true);
		pGPST.actulizacionPeridodica(milisPeriodo);
	}

	
	/** @return milisegundos del periodo de actualización */
	public int getMilisPeriodo() {
		return milisPeriodo;
	}

	/** @param milisPeriodo milisegundo a utilizar en la actualización. Deben ser >=0 */ 
	public void setMilisPeriodo(int milisPeriodo) {
		if(milisPeriodo<=0)
			throw new IllegalArgumentException("Milisegundos de actulización "+milisPeriodo+" deben ser >=0");
		this.milisPeriodo = milisPeriodo;
		pGPST.actulizacionPeridodica(milisPeriodo);
	}

	/**
	 * Se conecta a gps y abre ventana
	 * @param args puerto serial donde encontrar GPS
	 */
	public static void main(String[] args) {
		String[] puertos;
		if(args==null || args.length<1) {
			//no se han pasado argumentos, pedimos los puertos interactivamente
			String[] titulos={"GPS"};			
			puertos=new EligeSerial(titulos).getPuertos();
			if(puertos==null) {
				System.err.println("No se asignaron los puertos seriales");
				System.exit(1);
			}
		} else puertos=args;
		
		try {
			GPSConnectionTriumph gpsT=new GPSConnectionTriumph(puertos[0]);
			System.out.println("Esperamos por la posición de la base");
			if(gpsT.esperaCentroBase(20)) {
				System.out.println("Base en "+gpsT.posicionDeLaBase());
				gpsT.fijaCentro(gpsT.posicionDeLaBase());
			} else
				System.err.println("NO se consiguió la posición de la base");
			System.out.println("Comenzamos envío periódico desde GPS");
//			gpsT.comienzaEnvioPeriodico("%em%em,,{nmea/GGA,nmea/GSA,nmea/GST,nmea/VTG}:10\n");
			gpsT.comienzaEnvioPeriodico();

			VentanaGPSTriumph pc = new VentanaGPSTriumph(gpsT);
			pc.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
			
			VentanaLoggers vl=new VentanaLoggers();
			vl.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

			
			while (true){		
				Thread.sleep(500); 	
			}
		}
		catch (Exception e) {
			e.printStackTrace();
		}

	}

}
