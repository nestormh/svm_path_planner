package sibtra.util;

import javax.swing.JLabel;

/**
 * Para unificar todas las etiquetas de datos que se actualizan
 * @author alberto
 *
 */
public class LabelDato extends JLabel  {

	protected LabelDato copiaResumen=null;

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

	/** crea otro LabelDato con la mismas características que el actual */
	public LabelDato copiaParaResumen() {
		copiaResumen=new LabelDato(getText());
		copiaResumen.copiaResumen=copiaResumen; //apunta a si mismo para indicar que esta en resumen
		copiaResumen.setEnabled(isEnabled());
		return copiaResumen;
	}
	
	public void borradaCopiaResumen() {
		copiaResumen=null;
	}
	
	public boolean tieneCopiaParaResumen() {
		return copiaResumen!=null;
	}
	
	public boolean esCopiaParaResumen() {
		return copiaResumen==this;
	}
}
