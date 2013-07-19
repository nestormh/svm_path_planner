package sibtra.util;

import java.awt.Color;
import java.awt.event.ActionEvent;
import java.util.Vector;

import javax.swing.JCheckBox;

import sibtra.gps.Trayectoria;


/**
 * Panel que usa {@link PanelMapa} para mostrar una trayectoria la y posición del coche.
 * Por ahora es estático y no admite añadir un punto, sólo cambiar toda la trayectoria.
 * @author alberto
 */
@SuppressWarnings("serial")
public class PanelMuestraTrayectoria extends PanelMuestraVariasTrayectorias {
		
	/** Para marcar si se quiere mostrar los puntos */
	protected JCheckBox jcbMostrarPuntos;

	/** Para marcar si se quiere mostrar el rumbo */
	protected JCheckBox jcbMostrarRumbo;
	
    /**
     * Constructor 
     */
	public PanelMuestraTrayectoria() {
		super();

		JCheckBox jcba;
		
		jcbMostrarPuntos=jcba=new JCheckBox("Puntos");
		jpSur.add(jcba);
		jcba.setEnabled(false);
		jcba.addActionListener(this);
		jcba.setSelected(false);

		jcbMostrarRumbo=jcba=new JCheckBox("Rumbo");
		jpSur.add(jcba);
		jcba.setEnabled(false);
		jcba.addActionListener(this);
		jcba.setSelected(false);
		
		//Si queremoa añadir algo al panel inferiro	
		//		jpSur.add(jcbEscalas);

	}
	
	/** Establece la trayectoria a representar, pero no actualiza el panel
	 * 
	 * @param tr debe tener al menos 2 columnas
	 */
	public void setTrayectoria(Trayectoria tr) {
		if(tr==null || tr.length()==0) {
			jcbMostrarPuntos.setEnabled(false);
			jcbMostrarRumbo.setEnabled(false);
			borraTrayectorias();
		} else {
			jcbMostrarPuntos.setEnabled(true);
			jcbMostrarRumbo.setEnabled(true);
			borraTrayectorias();
			añadeTrayectoria(tr, Color.YELLOW);
			jcbMostrarPuntos.setSelected(false);
			jcbMostrarRumbo.setSelected(false);
		}
	}
	
	/** @param im vector de indice de puntos a marcar, null para no marcar */
	public void setMarcados(Vector<Integer> im) {
		setMarcados(0, im);
	}

	public void actionPerformed(ActionEvent ae) {
		if(ae.getSource()==jcbMostrarPuntos && trays.size()>0)
			setPuntos(0, jcbMostrarPuntos.isSelected());
		if(ae.getSource()==jcbMostrarRumbo && trays.size()>0)
			setRumbo(0, jcbMostrarRumbo.isSelected());
		super.actionPerformed(ae);
	}


}
