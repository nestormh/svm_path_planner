package sibtra.gps;

import java.awt.BorderLayout;
import javax.swing.JFrame;

import sibtra.log.VentanaLoggers;
import sibtra.util.EligeSerial;

/**
 * Ventana con información del GPS Triumph. Aparte del panel GPSData 
 * tiene información especifica del GPS. 
 * Crea hilo para refrescarse periodicamente, NO utiliza los eventos del GPS.
 * @author alberto
 *
 */
public class VentanaGPSTriumph extends JFrame implements Runnable {

	private PanelGPSTriumph pGPST;
	
	private GPSConnectionTriumph gpsCT;
	private Thread ThreadPanel;

	/** Milisegundos del periodo de actualización */
	private long milisPeriodo=500;


	public VentanaGPSTriumph(GPSConnectionTriumph gct) {
		super("VentanaGPSTriumph");
		if(gct==null)
			throw new IllegalArgumentException("Conexion a GPS no puede ser null");
		gpsCT=gct;
		pGPST=new PanelGPSTriumph(gpsCT);
		add(pGPST,BorderLayout.CENTER);
		pack();
		setVisible(true);
		ThreadPanel = new Thread(this);
		ThreadPanel.start();
	}

	/** Función del thread para refrescar ventana */
	public void run() {
		while (true){
			setEnabled(true);
			//TODO el ultimo punto debería depender de lo seleccionado en MuestraGPSData
			pGPST.actualizaGPS(gpsCT.getPuntoActualTemporal());
			pGPST.repinta();
			try{Thread.sleep(milisPeriodo);} catch (Exception e) {}	
		}
	}

	
	/** @return milisegundos del periodo de actualización */
	public long getMilisPeriodo() {
		return milisPeriodo;
	}

	/** @param milisPeriodo milisegundo a utilizar en la actualización. Deben ser >=0 */ 
	public void setMilisPeriodo(long milisPeriodo) {
		if(milisPeriodo<=0)
			throw new IllegalArgumentException("Milisegundos de actulización "+milisPeriodo+" deben ser >=0");
		this.milisPeriodo = milisPeriodo;
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
