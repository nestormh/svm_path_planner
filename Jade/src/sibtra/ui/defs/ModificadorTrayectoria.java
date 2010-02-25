/**
 * 
 */
package sibtra.ui.defs;

import sibtra.gps.Trayectoria;

/**
 * Lo deben implementar aquellos submodulos que vayan a realizar
 * modificaciones de la trayectoria inicial.
 * <br>
 * Típicamente el {@link Motor}, en cada iteración, le pedirá una trayectoria más 
 * actualizada a través de {@link #getTrayectoriaActual()}
 * <br>
 * La trayectoria inicial se le fijará con {@link #setTrayectoriaInicial(Trayectoria)}
 * 
 * @author alberto
 *
 */
public interface ModificadorTrayectoria extends SubModulo {

	/** El {@link Motor} le indica la trayectoria inicial sobre la que trabajar */
	void setTrayectoriaInicial(Trayectoria tra);
	
	/** EL {@link Motor}, en cada ciclo, le pedirá la trayectoria más actualizada.
	 * Tendrá las modificaciones respecto a la inicial si el módulo lo considera 
	 * necesario.
	 * <strong>Devolverá null si la trayectoria no ha cambiado con respecto a la llamanda anterior</strong>
	 */
	Trayectoria getTrayectoriaActual();
}
