/**
 * 
 */
package sibtra.ui.defs;


/**
 * Lo debe implentar aquel {@link SubModulo} que sea capaz de generar consignas de dirección.
 * 
 * @author alberto
 *
 */
public interface CalculoDireccion extends SubModulo {
	
	/** @return la consigna de velocida calculada */
	public double getConsignaDireccion();


}
