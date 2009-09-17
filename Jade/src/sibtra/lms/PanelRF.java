/**
 * 
 */
package sibtra.lms;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JLabel;
import javax.swing.JProgressBar;
import javax.swing.Timer;

import sibtra.util.PanelFlow;

/**
 * Panel que muestra los barridos recibidos del RF.
 * Implementa la auto-actualización.
 * @author alberto
 *
 */
public class PanelRF extends PanelMuestraBarrido {
	
	ManejaLMS manLMS;
	private sibtra.lms.PanelRF.ThreadActulizacion thActuliza;
	private BarridoAngular barrAct;
	private JProgressBar jpbTRF;
	private long deltaT;
	private JLabel jlTiempo;
	
	public PanelRF(ManejaLMS manLms) {
		super((short)80);
		manLMS=manLms;
		try {
			System.out.println(getClass().getName()+": Pedimos las zonas para establecerlas en el panel");
			setZona(manLMS.recibeZona((byte)0, true));
			setZona(manLMS.recibeZona((byte)1, true));
			setZona(manLMS.recibeZona((byte)2, true));
				
		} catch (LMSException e){
			System.err.println(getClass().getName()+": Problema al pedir las zonas:"+e.getMessage());
		}

		{
			PanelFlow jpTiempo=new PanelFlow();
			//barra de progreso para tiempo RF
			jpbTRF=new JProgressBar(0,100);
			jpbTRF.setOrientation(JProgressBar.HORIZONTAL);
			jpbTRF.setValue(0);
			jlTiempo=new JLabel("Tiempo RF= #### ms");
			jpTiempo.add(jlTiempo);
			jpTiempo.add(jpbTRF);

			add(jpTiempo); //siguiente línea de la caja
		}
		thActuliza=new ThreadActulizacion();
		thActuliza.start(); //arranca suspendido
		
		//creamos el action listener y el timer
		ActionListener taskPerformer = new ActionListener() {
			public void actionPerformed(ActionEvent evt) {
				actualiza();
			}
		};
		new Timer(200, taskPerformer).start();

		

	}
	
	/** Thread pendiente de la recepción de barridos por el LMS.
	 *  Se puede suspender y activar de manera correcta 
	 */
	class ThreadActulizacion extends Thread {
		private boolean suspendido=true;
		
		public synchronized void activar() {
			if(suspendido) {
				suspendido=false;
				notify();
			}
		}
		
		public synchronized void suspender() {
			if(!suspendido) {
				suspendido=true;
			}
		}
		
		public synchronized  boolean isSuspendido() {
			return suspendido;
		}
		

		public void run() {
			while(true) {
				actualizaBarrido();
				try {
					synchronized (this) {
						while (suspendido) wait(); 
					}
				} catch (InterruptedException e) {	}
			}
		}
	}
	
	/** Bloquea el threada hasta que llegue un nuevo barrido al LMS y actualiza el panel */
	public void actualizaBarrido() {
		long t0=System.currentTimeMillis();
		BarridoAngular nb=manLMS.esperaNuevoBarrido(barrAct);
		if(nb==barrAct) {
			//no se debe dar nunca
			System.err.println(this.getClass().getName()+": Es el mismo barrido");
			return; 
		}
		deltaT=System.currentTimeMillis()-t0;
//		System.out.println("Deta t="+deltaT);
		setBarrido(nb);
		actualiza();
		barrAct=nb; //para saber si cambia
	}

	/** Actualiza la etiqueta de tiempo y barra de progreso, o desactivoa si LSM no emitiendo continuo */
	public void actualiza() {
		if(!thActuliza.isSuspendido() && manLMS.isEnvioContinuo() ) {
			jpbTRF.setValue((int) deltaT);
			//TODO que sea una media ya que el valor varía mucho
			jlTiempo.setText(String.format("Tiempo RF= %5d ms", deltaT));
			jpbTRF.setEnabled(true);
			jlTiempo.setEnabled(true);
		} else {
			//no hay envios continuos
			jpbTRF.setEnabled(false);
			jlTiempo.setEnabled(false);
		}
		super.actualiza();
	}

	/** arranca thread de actualización coninua */ 
	public void actualizacionContinua() {
		thActuliza.activar();
	}
	
	/** susepende el thread de actualización continua */
	public void actualizacionContinuaParar() {
		thActuliza.suspender();
	}
}
