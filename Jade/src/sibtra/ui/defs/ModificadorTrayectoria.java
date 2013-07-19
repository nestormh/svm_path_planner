/**
 * 
 */
package sibtra.ui.defs;

import sibtra.gps.Trayectoria;

/**
 * Lo deben implementar aquellos submodulos que vayan a realizar
 * modificaciones de la trayectoria inicial.
 * <br>
 * La trayectoria inicial se le fijará con {@link #setTrayectoriaInicial(Trayectoria)}
 * <br>
 * Cuando tenga una trayectoria nueva se la comunica al {@link Motor} a través 
 * de {@link Motor#nuevaTrayectoria(Trayectoria) nuevaTrayectoria(Trayectoria)}. 
 * Éste deberá comunicarsela a todos los {@link SubModuloUsaTrayectoria} que tenga apuntados.
 * <br>
 * Dada esta forma de funcionamiento deberá tener un {@link Thread} donde se hará los calculos y se
 * comunicará la nueva trayectoria. Este thread trabajará concurrentemente con el del Motor.
 * <br> 
 * Éste le avisa cuando actúa y cuando se para invocando {@link #actuar()} y {@link #parar()} respectivamente.
 * 
 * @author alberto
 *
 */
public interface ModificadorTrayectoria extends SubModulo {

	/** El {@link Motor} le indica la trayectoria inicial sobre la que trabajar */
	void setTrayectoriaInicial(Trayectoria tra);
		
	/** El {@link Motor} le comunicará cuando comienza a actuar*/
	void actuar();
	
	/** El {@link Motor} le comunicará cuando para de actuar */
	void parar();
}
