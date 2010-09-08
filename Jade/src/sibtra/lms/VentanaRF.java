package sibtra.lms;

import javax.swing.JFrame;

import sibtra.util.EligeSerial;

/**
 * Muestra los barridos recibidos del RF usando {@link PanelRF}
 * @author alberto
 *
 */
public class VentanaRF extends JFrame {

	private ManejaLMS221 manLMS;
	private PanelRF panRF;
	
	public VentanaRF(String ptoRF) {
		super("Ventana RF");
		System.out.println("Conectamos a RF en: "+ptoRF);
		try { 		
			manLMS=new ManejaLMS221(ptoRF);
			manLMS.setDistanciaMaxima(80);
			manLMS.CambiaAModo25(); 

			//creamos panel muestra barrido
			panRF=new PanelRF(manLMS);
			add(panRF);

			manLMS.pideBarridoContinuo((short)0,(short) 180, (short)1);
		} catch (LMSException e) {
			System.err.println("No fue posible conectar o configurar RF: "+e.getMessage());
			System.exit(1);
		}

		
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		pack();
		setVisible(true);
		setBounds(0, 384, 1000, 700);
		panRF.actualizacionContinua();
		
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
