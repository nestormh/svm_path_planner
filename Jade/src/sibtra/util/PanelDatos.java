package sibtra.util;

import java.awt.Color;
import java.awt.FlowLayout;
import java.awt.Font;
import java.util.Vector;

import javax.swing.BorderFactory;
import javax.swing.JComponent;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;
import javax.swing.border.Border;

/**
 * Es {@link JPanel} que gestiona LabelDatos permitiendo añadirlos y
 * actulizarlos.
 * Clase padre de distintos paneles que muestran datos. 
 * @author alberto
 *
 */
public class PanelDatos extends JPanel {
	/** Fuente a usar en las LabelDatos */
	private Font Grande;
	/** Borde que rodea las etiquetas*/
	private Border blackline = BorderFactory.createLineBorder(Color.black);
	/** Panel por defecto donde se añadirán las etiquetas */
	private JPanel jpPorDefecto=null;
	
	/** Las etiquetas que se añaden y que deben actualizarse */
	private Vector<LabelDato> vecLabels=new Vector<LabelDato>();
	
	/** Constructor sin definir panel por defecto */
	public PanelDatos() {
		Grande = getFont().deriveFont(20.0f);
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
		lda.setFont(Grande);
		añadeAPanel((JComponent)lda, Titulo, panAñadir);		
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
	
	/** Añade componenete (no gestionado) a panel con borde y título */
	public void añadeAPanel(JComponent jcmp, String Titulo) {
		if(jpPorDefecto==null)
			throw new IllegalArgumentException("No esta definido panel por defecto");
		añadeAPanel(jcmp, Titulo, jpPorDefecto);
	}
	
	/** Añade componente (no gestionado) a panel indicado con borde y título */
	public void añadeAPanel(JComponent jcmp, String Titulo, JPanel panAñadir) {
		jcmp.setBorder(BorderFactory.createTitledBorder(
				blackline, Titulo));
		jcmp.setEnabled(false);
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

	/** programamos la actualizacion del panel */
	public void repinta() {
		SwingUtilities.invokeLater(new Runnable() {
			public void run() {
				repaint();
			}
		});
	}
	
}
