/**
 * 
 */
package sibtra.ui.defs;

/**
 * Lo deben implentar los submódulos que dependan de {@link Motor}.
 * <br>
 * Por ahora, necesitan conocer el {@link Motor} para poder pedirle la trayectoria si la necesitan
 * a través de {@link Motor#apuntaNecesitaTrayectoria(SubModuloUsaTrayectoria)}
 * 
 * @author alberto
 *
 */
public interface SubModulo extends Modulo {

	/** Establece el {@link Motor} */
	public void setMotor(Motor mtr);
}
