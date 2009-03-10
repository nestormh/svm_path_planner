package sibtra.controlcarro;

import java.awt.BorderLayout;
import javax.swing.JFrame;

import sibtra.log.VentanaLoggers;
import sibtra.util.EligeSerial;

/** 
 * Ventana para la monitorización de la información recibida del coche a través
 * del {@link ControlCarro}. 
 * Hace uso del {@link PanelCarro} y añade tread para hacer actualización periódica
 * @author alberto,jonay
 *
 */

@SuppressWarnings("serial")
public class VentanaCoche extends JFrame implements Runnable {

	private Thread ThreadPanel;
	private PanelCarro panCarro;
	/** Milisegundos del periodo de actualización */
	private long milisPeriodo=500;

	
	public void run() {
		while (true){
			setEnabled(true);
			panCarro.actualizaCarro();
			panCarro.repinta();
			try{Thread.sleep(milisPeriodo);} catch (Exception e) {}	
		}
	}

	/**
	 * Costructor crea el {@link PanelCarro} y el thread de actualización
	 * @param cc carro del que leer los datos
	 */
	public VentanaCoche(ControlCarro cc) {
		super("Control Carro");
		if(cc==null) 
			throw new IllegalArgumentException("Control de carro pasado no puede ser null");
		panCarro=new PanelCarro(cc);

		add(panCarro,BorderLayout.CENTER);
		pack();
		setVisible(true);
		ThreadPanel = new Thread(this);
		ThreadPanel.start();

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

	/** Crea la ventana con usando la serial pasada como primer parámetro. Si no se pasa
	 * se pide interactivamente.
	 */
	public static void main(String[] args) {
		String[] puertos;
		if(args==null || args.length<1) {
			//no se han pasado argumentos, pedimos los puertos interactivamente
			String[] titulos={"Carro"};			
			puertos=new EligeSerial(titulos).getPuertos();
			if(puertos==null) {
				System.err.println("No se asignaron los puertos seriales");
				System.exit(1);
			}
		} else puertos=args;
		
		ControlCarro contCarro=new ControlCarro(puertos[0]);		
		VentanaCoche pc = new VentanaCoche(contCarro);
		pc.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

		
		VentanaLoggers vl=new VentanaLoggers();
		vl.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);


		while (true){		
			try{Thread.sleep(500);} catch (Exception e) {}	
		}
	}


	
}
