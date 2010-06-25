package sibtra.controlcarro;

import java.awt.BorderLayout;
import java.awt.event.ActionEvent;

import javax.swing.AbstractAction;
import javax.swing.JButton;
import javax.swing.JFrame;

import sibtra.log.VentanaLoggers;
import sibtra.util.EligeSerial;

/** 
 * Ventana para la monitorización de la información recibida del coche a través
 * del {@link ControlCarro}. 
 * Hace uso del {@link PanelCarro} activa su auto-actualización.
 * @author alberto,jonay
 *
 */

@SuppressWarnings("serial")
public class VentanaCoche extends JFrame {

	private PanelCarro panCarro;
	/** Milisegundos del periodo de actualización */
	private int milisPeriodo=500;


	/**
	 * Costructor crea el {@link PanelCarro} y el thread de actualización
	 * @param cc carro del que leer los datos
	 */
	public VentanaCoche(ControlCarro cc) {
		super("Control Carro");
		if(cc==null) 
			throw new IllegalArgumentException("Control de carro pasado no puede ser null");
		panCarro=new PanelCarro(cc) {
			final class accFijaVel extends AbstractAction {
				double velAplicar=0.0;
				public accFijaVel(double vel) {
					super(vel+" ");
					velAplicar=vel;
				}
				public void actionPerformed(ActionEvent ev) {
					contCarro.setConsignaAvanceMS(velAplicar);
				}
			}
			{ //inicializador de la instancia, reemplaza al constructor
				añadeAPanel(new JButton(new accFijaVel(0)), "Cons.V");
				añadeAPanel(new JButton(new accFijaVel(1)), "Cons.V");
				añadeAPanel(new JButton(new accFijaVel(2)), "Cons.V");
				añadeAPanel(new JButton(new accFijaVel(3)), "Cons.V");
			}
		};

		add(panCarro,BorderLayout.CENTER);
		panCarro.actulizacionPeridodica(milisPeriodo);
		pack();
		setVisible(true);
	}
	
	
	/** @return milisegundos del periodo de actualización */
	public long getMilisPeriodo() {
		return milisPeriodo;
	}

	/** @param milisPeriodo milisegundo a utilizar en la actualización. Deben ser >=0 */ 
	public void setMilisPeriodo(int milisPeriodo) {
		if(milisPeriodo<=0)
			throw new IllegalArgumentException("Milisegundos de actulización "+milisPeriodo+" deben ser >=0");
		this.milisPeriodo=milisPeriodo;
		panCarro.actulizacionPeridodica(milisPeriodo);
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
