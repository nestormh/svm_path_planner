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
 * Cada vez que varíe la trayectoria diruante la ejecución, tipicamente porque el {@link ModificadorTrayectoria} he hecho
 * cambios,
 * serán avisados mediante {@link #setTrayectoriaModificada(Trayectoria)}.
 * 
 * @author alberto
 *
 */
public interface SubModuloUsaTrayectoria extends SubModulo {

	/** El {@link Motor} le indica la trayectoria inicial sobre la que trabajar */
	void setTrayectoriaInicial(Trayectoria tra);
	
	/** A través de este método, el {@link Motor} avisará al módulo, durante la ejecución si la trayectoria se ha modificado*/
	void setTrayectoriaModificada(Trayectoria tra);
	
}
