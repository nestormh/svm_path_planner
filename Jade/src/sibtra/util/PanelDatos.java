package sibtra.util;

import java.awt.Color;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.Vector;

import javax.swing.BorderFactory;
import javax.swing.JComponent;
import javax.swing.JLabel;
import javax.swing.JMenuItem;
import javax.swing.JPanel;
import javax.swing.JPopupMenu;
import javax.swing.SwingUtilities;
import javax.swing.Timer;
import javax.swing.border.Border;
import javax.swing.border.TitledBorder;

/**
 * Es {@link JPanel} que gestiona LabelDatos permitiendo añadirlos y
 * actulizarlos.
 * Clase padre de distintos paneles que muestran datos. 
 * @author alberto
 *
 */
public class PanelDatos extends JPanel {
	/** Fuente a usar en las LabelDatos */
	protected Font fuenteLabel;
	/** Fuente a usar en el titulo del borde */
	protected Font fuenteTitulo;
	/** Borde que rodea las etiquetas*/
	private Border blackline = BorderFactory.createLineBorder(Color.black);
	/** Panel por defecto donde se añadirán las etiquetas */
	private JPanel jpPorDefecto=null;
	
	/** Las etiquetas que se añaden y que deben actualizarse */
	private Vector<LabelDato> vecLabels=new Vector<LabelDato>();
	

	private Timer timerActulizacion;
	private JPanel panelResumen=null;

	/** Constructor sin definir panel por defecto */
	public PanelDatos() {
		fuenteLabel = getFont().deriveFont(18.0f);
		fuenteTitulo = getFont().deriveFont(12.0f);
		jpPorDefecto=this; //panel por defecto es él mismo
	}

	/** Constructor pasandole el layout */
	public PanelDatos(FlowLayout layout) {
		this();
		setLayout(layout);
	}

	/** 
	 * Añade lda a la lista de etiquetas gestionadas, pero sin añadirla
	 * a ningún panel.
	 * @param lda etiqueta, no puede ser NULL
	 */
	public void añadeLabel(LabelDato lda) {
		if(lda==null)
			throw new IllegalArgumentException("Label a añadir no puede ser NULL");
		vecLabels.add(lda);
	}
	
	/** 
	 * Elimina lda a la lista de etiquetas gestionadas {@link #vecLabels}, pero sin quitarla
	 * de ningún panel.
	 * @param lda etiqueta
	 */
	public void quitaLabel(LabelDato lda) {
		vecLabels.removeElement(lda);
		for(LabelDato la: vecLabels) {
			if(la.copiaResumen==lda)
				la.borradaCopiaResumen();
		}
			
	}
	
	/** Asigna el panel por defecto */
	public void setPanelPorDefecto(JPanel panPorDefecto) {
		jpPorDefecto=panPorDefecto;
	}
	
	/** @return el panel por defecto */
	public JPanel getPanelPorDefecto() {
		return jpPorDefecto;
	}

	/**
	 * Funcion para añadir etiqueta con todas las configuraciones por defecto
	 * @param lda etiqueta a añadir
	 * @param Titulo titulo adjunto
	 * @param panAñadir panel donde añadir la etiqueta
	 */
	public void añadeAPanel(LabelDato lda,String Titulo, JPanel panAñadir) {
		añadeLabel(lda);
		lda.setHorizontalAlignment(JLabel.CENTER);
		lda.setFont(fuenteLabel);
    	// ponemos popups
    	lda.addMouseListener(new MouseAdapter() {
    		public void mousePressed(MouseEvent me)
    		{
    			muestraPopUp(me);
    		}

    		public void mouseReleased(MouseEvent me)
    		{
    			muestraPopUp(me);
    		}
    	});

		añadeAPanel((JComponent)lda, Titulo, false, panAñadir);		
	}
	
	protected void muestraPopUp(MouseEvent me) {
		final LabelDato lda;
		if(panelResumen!=null && me.isPopupTrigger()) {
			lda=(LabelDato)me.getSource();
			JPopupMenu popup = new JPopupMenu();
			if(!lda.esCopiaParaResumen()) {
				//está en un panel normal
				JMenuItem item = new JMenuItem("Poner en Resumen");
				popup.add(item);
				if(!lda.tieneCopiaParaResumen()) {
					//aún no tiene copia
					item.addActionListener(new ActionListener() {
						public void actionPerformed(ActionEvent e)
						{
							String Titulo=((TitledBorder)lda.getBorder()).getTitle();
							LabelDato lres=lda.copiaParaResumen();
//							System.out.println("Se eligió poner en resumen de "+lres);
							añadeAPanel(lres, Titulo, panelResumen);
							panelResumen.repaint();
						}
					});
				} else {
					//ya tiene copia, debe estar deshabilitado
					item.setEnabled(false);
				}
			} else {
				// es la copia en resumen
				JMenuItem item = new JMenuItem("Quitar de Resumen");
				popup.add(item);
				item.addActionListener(new ActionListener() {
					public void actionPerformed(ActionEvent e)
					{
						quitaLabel(lda);
						panelResumen.remove(lda);
//						System.out.println("Se quitó "+lda);
						panelResumen.repaint();
					}
				});
			}
			popup.show(me.getComponent(), me.getX(), me.getY());
		}
	}

	/** Añade etiqueta con todas las configuraciones por defecto en el
	 * panel por defecto.
	 * @param lda etiqueta a añadir
	 * @param Titulo titulo adjunto
	 */
	public void añadeAPanel(LabelDato lda,String Titulo) {
		if(jpPorDefecto==null)
			throw new IllegalArgumentException("No esta definido panel por defecto");
		añadeAPanel(lda, Titulo, jpPorDefecto);
	}
	
	/** Añade componenete (no gestionado) a panel con borde y título por defecto habilitado */
	public void añadeAPanel(JComponent jcmp, String Titulo) {
		añadeAPanel(jcmp, Titulo, true);
	}
	
	/** Añade componenete (no gestionado) a panel con borde y título, indicando si está habilitado */
	public void añadeAPanel(JComponent jcmp, String Titulo, boolean enabled) {
		if(jpPorDefecto==null)
			throw new IllegalArgumentException("No esta definido panel por defecto");
		añadeAPanel(jcmp, Titulo, enabled, jpPorDefecto);
	}
	
	/** Añade componente (no gestionado) a panel indicado con borde y título, indicando si está habilitado */
	public void añadeAPanel(JComponent jcmp, String Titulo, boolean enabled, JPanel panAñadir) {
		jcmp.setBorder(BorderFactory.createTitledBorder(
				blackline, Titulo,TitledBorder.LEFT,TitledBorder.TOP,fuenteTitulo));
		jcmp.setEnabled(enabled);
		panAñadir.add(jcmp);
	}

	/** Actualiza las etiquetas del panel
	 * @param nuevoObj objeto del que sacarán información las etiquetas
	 */
	public void actualizaDatos(Object nuevoObj) {
		boolean hayDato=(nuevoObj!=null);
		//atualizamos etiquetas en array
		for(int i=0; i<vecLabels.size(); i++)
			vecLabels.elementAt(i).Actualiza(nuevoObj,hayDato);
	}

	/** Método que se invocará en la {@link #actulizacionPeridodica(int)}.
	 * En esta clase llama al {@link #repaint()}. 
	 * Las clases hijas que quieran hacer uso de la actualización tredrá que modificarla
	 */
	protected void actualiza() {
		repaint();
	}
	
	/** Establece timer para la actulización periódica del panel.
	 * Si ya existe uno simplemente cambia el periodo.
	 * @param periodoMili periodo de actulización en milisegundos
	 */
	public void actulizacionPeridodica(int periodoMili) {
		if(timerActulizacion==null) {
			//creamos el action listener y el timer
			ActionListener taskPerformer = new ActionListener() {
				public void actionPerformed(ActionEvent evt) {
//					System.out.print("+");
					actualiza();
				}
			};
			timerActulizacion=new Timer(periodoMili, taskPerformer);
		} else 
			//basta con modificar el delay
			timerActulizacion.setDelay(periodoMili);
		timerActulizacion.start();
	}

	/** Detiene la acutualización periodica si existe alguna */
	public void actualizacionPeriodicaParar() {
		if(timerActulizacion==null) return;
		timerActulizacion.stop();
	}
	

	
	/** programamos la actualizacion del panel */
	public void repinta() {
		SwingUtilities.invokeLater(new Runnable() {
			public void run() {
				repaint();
			}
		});
	}
	

	public void setPanelResumen(JPanel panelResumen) {
		this.panelResumen=panelResumen;		
	}

}
