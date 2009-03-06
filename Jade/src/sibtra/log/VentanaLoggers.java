package sibtra.log;

import java.awt.BorderLayout;

import javax.swing.JFrame;

public class VentanaLoggers extends JFrame implements Runnable {
	
	private PanelLoggers panLog;
	
	private Thread ThreadPanel;

	/** Milisegundos del periodo de actualización */
	private long milisPeriodo=1000;

	public VentanaLoggers() {
		super("Loggers");
		panLog=new PanelLoggers();
		add(panLog,BorderLayout.CENTER);
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
			panLog.repinta();
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




}
