package sibtra.imu;

import java.awt.BorderLayout;

import javax.swing.JFrame;

import sibtra.controlcarro.PanelCarro;
import sibtra.log.VentanaLoggers;
import sibtra.util.EligeSerial;

/** 
 * Ventana para la monitorización de la información recibida de la IMU
 * a través de  {@link ConexionSerialIMU}. 
 * Hace uso del {@link PanelIMU} y programa su actulización periodica.
 * NO utiliza los eventos de recepción de la IMU.
 * @author alberto
 *
 */
@SuppressWarnings("serial")
public class VentanaIMU extends JFrame {

		
		private PanelIMU panIMU;
		
		/** Conexión a la IMU*/
		private ConexionSerialIMU conSerIMU;
		
		/** Milisegundos del periodo de actualización */
		private int milisPeriodo=500;

		
		public void run() {
			while (true){
				setEnabled(true);
				panIMU.actualizaAngulo(conSerIMU.getAngulo());
				panIMU.repinta();
				try{Thread.sleep(milisPeriodo);} catch (Exception e) {}	
			}
		}

		/**
		 * Costructor crea el {@link PanelIMU} y el thread de actualización
		 * @param csi conexión a la IMU
		 */
		public VentanaIMU(ConexionSerialIMU csi) {
			super("Angulos IMU");
			if(csi==null) 
				throw new IllegalArgumentException("Conexion serial IMU no puede ser null");
			conSerIMU=csi;
			panIMU=new PanelIMU(csi);

			add(panIMU,BorderLayout.CENTER);
			pack();
			setVisible(true);
			
			panIMU.actulizacionPeridodica(milisPeriodo);
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
			panIMU.actulizacionPeridodica(milisPeriodo);
		}

		/** Crea la ventana con usando la serial pasada como primer parámetro. Si no se pasa
		 * se pide interactivamente.
		 */
		public static void main(String[] args) {
			String[] puertos;
			if(args==null || args.length<1) {
				//no se han pasado argumentos, pedimos los puertos interactivamente
				String[] titulos={"IMU"};			
				puertos=new EligeSerial(titulos).getPuertos();
				if(puertos==null) {
					System.err.println("No se asignaron los puertos seriales");
					System.exit(1);
				}
			} else puertos=args;
			
			ConexionSerialIMU csi=new ConexionSerialIMU();
			csi.ConectaPuerto(puertos[0]);
			VentanaIMU vi = new VentanaIMU(csi);
			vi.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
			
			VentanaLoggers vl=new VentanaLoggers();
			vl.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

			while (true){		
				try{Thread.sleep(500);} catch (Exception e) {}	
			}
		}

}
