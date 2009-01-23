package sibtra.util;

import javax.swing.JLabel;

/**
 * Para unificar todas las etiquetas de datos que se actualizan
 * @author alberto
 *
 */
public class LabelDato extends JLabel {

	/**
	 * Ponemos texto y quedamos deshabilitada
	 * @param textoInicial
	 */
	public LabelDato(String textoInicial) {
		setText(textoInicial);
		setEnabled(false);
	}
	
	/**
	 * Activa etiqueta si hayCambio. 
	 * Este método se deberá modificar por los hijos si hacen presentación especial 
	 * @param dato En la forma general no se usa
	 * @param hayCambio
	 */
	public void Actualiza(Object dato, boolean hayCambio){
		setEnabled(hayCambio);
	}
}
