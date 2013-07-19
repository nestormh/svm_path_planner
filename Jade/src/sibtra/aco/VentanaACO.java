/**
 * 
 */
package sibtra.aco;

import java.awt.BorderLayout;

import javax.swing.JFrame;

import sibtra.imu.VentanaIMU;
import sibtra.log.VentanaLoggers;

/**
 * @author alberto
 *
 */
@SuppressWarnings("serial")
public class VentanaACO extends JFrame {
	
	private PanelACO panACO;

	/** Milisegundos del periodo de actualización */
	private int milisPeriodo=500;

	
//	public void run() {
//		while (true){
//			setEnabled(true);
//			panACO.actualizaDatos();
//			panACO.repinta();
//			try{Thread.sleep(milisPeriodo);} catch (Exception e) {}	
//		}
//	}

	
	public VentanaACO() {
		super("Datos ACO");
		panACO=new PanelACO();
		add(panACO,BorderLayout.CENTER);
		pack();
		setVisible(true);
		
		panACO.actulizacionPeridodica(milisPeriodo);
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
		panACO.actulizacionPeridodica(milisPeriodo);
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		VentanaACO va = new VentanaACO();
		va.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		
		VentanaLoggers vl=new VentanaLoggers();
		vl.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

		while (true){		
			try{Thread.sleep(500);} catch (Exception e) {}	
		}
	}

}
