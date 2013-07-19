package sibtra.lms;

import java.awt.BorderLayout;
import java.awt.event.WindowEvent;
import java.awt.event.WindowListener;

import javax.swing.JFrame;

import sibtra.util.EligeSerial;
import sibtra.util.ThreadSupendible;


/** Aplicación para mostrar los barridos.
 * Sólo se actualiza mientras el cursor esté sobre la ventana.
 */
@SuppressWarnings("serial")
public class VentanaMuestraBarridos extends JFrame implements WindowListener {

	private ManejaLMS221 manLMS;
	private PanelMuestraBarrido pmb;
	/** contendrá el último barrido recibido del LMS */
	private BarridoAngular barrAct=null;
	private ThreadSupendible thActuliza;
	public VentanaMuestraBarridos(String ptoRF) {
		super("Muestra Barrido");
		//Conectamos a RF
		try { 		
			manLMS=new ManejaLMS221(ptoRF);
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
		
		thActuliza=new ThreadSupendible() {
			@Override
			protected void accion() {
				actualizaBarrido();
			}			
		};
//		thActuliza=new ThreadActualizacion();
//		thActuliza.start(); //arranca suspendido
		
		addWindowListener(this);
		setDefaultCloseOperation(JFrame.DO_NOTHING_ON_CLOSE);
		pack();
		setVisible(true);
        setBounds(0, 384, 1024, 742);

	}

//	class ThreadActualizacion extends ThreadSupendible {
//		@Override
//		protected void accion() {
//			actualizaBarrido();
//		}
//	}
	
	public void actualizaBarrido() {
		long t0=System.currentTimeMillis();
		BarridoAngular nb=manLMS.esperaNuevoBarrido(barrAct);
		if(nb==barrAct) {
			System.out.println("Es el mismo");
			return; 
		}
		long t1=System.currentTimeMillis();
		System.out.println("Deta t="+(t1-t0));
		pmb.setBarrido(nb);
		pmb.actualiza();
		barrAct=nb; //para saber si cambia
	}

	public void windowActivated(WindowEvent arg0) {
		System.out.println("windowActivated");
		windowDeiconified(arg0);
	}

	public void windowClosed(WindowEvent arg0) {
		System.out.println("windowClosed");
		System.exit(0);		
	}

	public void windowClosing(WindowEvent arg0) {
		System.out.println("windowClosing");
		windowIconified(arg0); //lo mismo que al iconificar
		setVisible(false); //la ocultamos
		System.exit(0); //no se llega a	windowClosed
	}

	public void windowDeactivated(WindowEvent arg0) {
		System.out.println("windowDeactivated");
		windowIconified(arg0);
	}

	public void windowDeiconified(WindowEvent arg0) {
		System.out.println("windowDeiconified");
		windowOpened(arg0); //lo mismo que si se abriera
	}

	public void windowIconified(WindowEvent arg0) {
		System.out.println("windowIconified");
		//solicitamos la paradad del envio continuo
		try {
			manLMS.pidePararContinuo();
		} catch (LMSException e) {
			System.err.println("Problema al parar continuo: "+e.getMessage());
		}
		thActuliza.suspender();
	}

	/** Solicitamos envío continuo y arrancamos thread de actualización */
	public void windowOpened(WindowEvent arg0) {
		System.out.println("windowOpened");
		try {
			manLMS.pideBarridoContinuo((short)0, (short)180, (short)1);
			thActuliza.activar();
			System.out.println("Comenzamos th de actualización");
		} catch (LMSException e) {
			System.err.println("No fue posible iniciar envío continuo:"+e.getMessage());
		}
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
		
	}


}
