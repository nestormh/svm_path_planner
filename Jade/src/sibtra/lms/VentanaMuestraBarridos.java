package sibtra.lms;

import java.awt.BorderLayout;

import javax.swing.JFrame;

import sibtra.util.EligeSerial;


/** Aplicación para mostrar los barridos */
@SuppressWarnings("serial")
public class VentanaMuestraBarridos extends JFrame {

	private ManejaLMS manLMS;
	private PanelMuestraBarrido pmb;
	/** contendrá el último barrido recibido del LMS */
	private BarridoAngular barrAct=null;
	public VentanaMuestraBarridos(String ptoRF) {
		super("Muestra Barrido");
		//Conectamos a RF
		try { 		
			manLMS=new ManejaLMS(ptoRF);
			manLMS.setDistanciaMaxima(80);
			manLMS.CambiaAModo25(); 
		} catch (LMSException e) {
			System.err.println("No fue posible conectar o configurar RF: "+e.getMessage());
			System.exit(1);
		}

		//creamos panel muestra barrido
		pmb=new PanelMuestraBarrido((short) 80);
		add(pmb,BorderLayout.CENTER);
		
		//pedimos las zonas para establecerlas en el panel
		try {
			pmb.setZona(manLMS.recibeZona((byte)0, true));
			pmb.setZona(manLMS.recibeZona((byte)1, true));
			pmb.setZona(manLMS.recibeZona((byte)2, true));
				
		} catch (LMSException e){
			System.err.println("Problema al pedir las zonas:"+e.getMessage());
		}
		
		
		try {
			manLMS.pideBarridoContinuo((short)0, (short)180, (short)1);
		} catch (LMSException e) {
			System.err.println("No fue posible inidiciar envío continuo:"+e.getMessage());
			System.exit(1);
		}
		
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		pack();
		setVisible(true);
	}

	private void actulizaBarrido() {
		long t0=System.currentTimeMillis();
		BarridoAngular nb=manLMS.ultimoBarrido();
		if(nb==barrAct) {
//			System.out.println("Es el mismo");
			return; 
		}
		long t1=System.currentTimeMillis();
//		System.out.println("Deta t="+(t1-t0));
		pmb.setBarrido(nb);
		pmb.actualiza();
		barrAct=nb; //para saber si cambia
	}

	
	/**
	 * @param args nombre del puerto serie del LMS, si no se pide interactivamente
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

		VentanaMuestraBarridos mb=new VentanaMuestraBarridos(puertos[0]);
		
		while(true) {
			mb.actulizaBarrido();
			try { Thread.sleep(100); } catch (Exception e) {}
		}
		
		
	}

}
