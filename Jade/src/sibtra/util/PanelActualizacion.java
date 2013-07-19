package sibtra.util;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JPanel;
import javax.swing.SwingUtilities;
import javax.swing.Timer;

/**
 * Panel que se actuliza periodicamente llamando al metodo {@link #actualizaPeriodico()}.
 * Las clases hijas deben sobreescribir este método para haces sus actulizaciones.
 * 
 * @author alberto
 *
 */
@SuppressWarnings("serial")
public class PanelActualizacion extends JPanel {

	private Timer timerActulizacion;
	/** Perido por defecto para la actualizacion */
	private int milisActulizacion = 500;
	
	/** Si hay datos nuevos en la actualización periódica */
	protected boolean hayDatosNuevos = false;
	
	public PanelActualizacion() {
		super();
	}

	/** Método que se invocará en la {@link #actulizacionPeridodica(int)}.
	 * En esta clase llama al {@link #repaint()}. 
	 * Las clases hijas que quieran hacer uso de la actualización tredrá que modificarla
	 */
	protected void actualizaPeriodico() {
		repaint();
	}

	/** Establece timer para la actulización periódica del panel.
	 * Si ya existe uno simplemente cambia el periodo.
	 * @param periodoMili periodo de actulización en milisegundos
	 */
	public void actulizacionPeridodica(int periodoMili) {
			if(periodoMili<=0)
				throw new IllegalArgumentException("Milisegundos de actulización "+periodoMili+" deben ser >=0");
			actualizacionEsperaParar(); //paramos la de espera
			if(timerActulizacion==null) {
				//creamos el action listener y el timer
				ActionListener taskPerformer = new ActionListener() {
					public void actionPerformed(ActionEvent evt) {
	//					System.out.print("+");
						actualizaPeriodico();
					}
				};
				timerActulizacion=new Timer(periodoMili, taskPerformer);
			} else 
				//basta con modificar el delay
				timerActulizacion.setDelay(periodoMili);
			milisActulizacion=periodoMili;
			timerActulizacion.start();
		}

	/** invoca {@link #actulizacionPeridodica(int)} con el valor de {@link #milisActulizacion} (valor por defecto) */
	public void actulizacionPeridodica() {
		actulizacionPeridodica(milisActulizacion);
	}

	public int getMilisActualizacion() {
		return milisActulizacion;
	}

	/** Detiene la acutualización periodica si existe alguna */
	public void actualizacionPeriodicaParar() {
		if(timerActulizacion!=null)
			timerActulizacion.stop();
	}

	//Para la actualización por espera
	/** */
	protected ThreadSupendible thEspera=null;
	
	public void actualizacionEspera() {
		//paramos actualización periodica
		actualizacionPeriodicaParar();
		if(thEspera==null) {
			thEspera=new ThreadSupendible() {

				@Override
				protected void accion() {
					actualizacionEspera();
				}
			};
			thEspera.setName(getName()+"thEspera");
		}
		//activamos el th
		thEspera.activar();
	}

	/** Detiene la actualizacion por espera */
	public void actualizacionEsperaParar() {
		if(thEspera!=null)
			thEspera.suspender();
	}
	
	/** Método que deben implementar los hijos en el deben esperar por nuevos datos que presentar */
	protected void actualizaEspera() {
		repinta();
	}

	/** programamos la actualizacion del panel con {@link SwingUtilities#invokeLater(Runnable)} */
	public void repinta() {
		SwingUtilities.invokeLater(new Runnable() {
			public void run() {
				repaint();
			}
		});
	}

}