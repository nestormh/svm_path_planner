/**
 * 
 */
package sibtra.ui.defs;

import sibtra.gps.Trayectoria;

/**
 * Lo deben cumplir los {@link SubModulo} que necesitan la trayectoria para sus cálculos.
 * <br>
 * Deben apuntarse en el motor mediante {@link Motor#apuntaNecesitaTrayectoria(SubModuloUsaTrayectoria)}.
 * Cada vez que varíe la trayectoria inicial, tipicamente al comenzar una nueva actuación, serán 
 * avisados mediante {@link #setTrayectoriaInicial(Trayectoria)}.
 * <br>
 * Si desean trayectorias más actualizadas (modificadas por un {@link ModificadorTrayectoria} si está 
 * definido) deben invocar {@link Motor#getTrayectoriaActual()}.
 * 
 * 
 * @author alberto
 *
 */
public interface SubModuloUsaTrayectoria extends SubModulo {

	/** El {@link Motor} le indica la trayectoria inicial sobre la que trabajar */
	void setTrayectoriaInicial(Trayectoria tra);
	
}
