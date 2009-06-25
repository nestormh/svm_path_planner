package sibtra.lms;

import javax.swing.JFrame;

import sibtra.util.EligeSerial;

/**
 * Muestra los barridos recibidos del RF usando {@link PanelRF}
 * @author alberto
 *
 */
public class VentanaRF extends JFrame {

	private ManejaLMS manLMS;
	private PanelRF panRF;
	
	public VentanaRF(String ptoRF) {
		super("Ventana RF");
		//Conectamos a RF
		try { 		
			manLMS=new ManejaLMS(ptoRF);
			manLMS.setDistanciaMaxima(80);
			manLMS.CambiaAModo25(); 
			manLMS.pideBarridoContinuo((short)0,(short) 180, (short)1);
		} catch (LMSException e) {
			System.err.println("No fue posible conectar o configurar RF: "+e.getMessage());
			System.exit(1);
		}

		//creamos panel muestra barrido
		panRF=new PanelRF(manLMS);
		panRF.actualizacionContinua();
		add(panRF);
		
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		pack();
		setVisible(true);

	}
	/**
	 * @param args
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
		VentanaRF vrf=new VentanaRF(puertos[0]);

	}

}
