/**
 * 
 */
package sibtra.ui.defs;

import sibtra.gps.Trayectoria;
import sibtra.predictivo.Coche;
import sibtra.ui.VentanasMonitoriza;


/**
 * Este módulo será el que llevará el thread de ejución del control.
 * Es el encargado de actuar sobre el coche.
 * Debe recibir los siguientes {@link SubModulo}:
 * <ul>
 * <li> un {@link CalculoDireccion} a través de {@link #setCalculadorDireccion(CalculoDireccion)}: al que le pide en cada iteración la consigna del volante 
 * a través de {@link CalculoDireccion#getConsignaDireccion() getConsignaDireccion()}</li>
 * <li> un {@link CalculoVelocidad} a través de {@link #setCalculadorVelocidad(CalculoVelocidad)}: al que le pide en cada iteración la consigna de velocidad 
 * a través de {@link CalculoVelocidad#getConsignaVelocidad() getConsignaVelocidad()}</li>
 * <li> 0 o varios {@link DetectaObstaculos} a través de {@link #setDetectaObstaculos(DetectaObstaculos[])}: en cada iteración les pedirá la distancia al obstáculo
 * más cercano detectado a través de {@link DetectaObstaculos#getDistanciaLibre() getDistanciaLibre()}.</li>
 * <li> 0 ó 1 {@link ModificadorTrayectoria} a través de {@link #setModificadorTrayectoria(ModificadorTrayectoria)}: 
 * cuando éste módulo tenga una nueva trayectoria invocará {@link #nuevaTrayectoria(Trayectoria)}. El motor debe limitarse
 * a invocar el {@link SubModuloUsaTrayectoria#setTrayectoriaModificada(Trayectoria) setTrayectoriaModificada(Trayectoria)} 
 * de los submódulos que hayan apuntado
 * mediante {@link #apuntaNecesitaTrayectoria(SubModuloUsaTrayectoria)}.
 * De esta manera es el módulo {@link ModificadorTrayectoria} el que toma la iniciativa y
 * , asincronamente (debe tener un thread propio), avisa a los usuarios . 
 * Por ello es necesario que dichos módulos <strong>gestionen correctamente el posible caracter simultáneo (desde distintos
 * threads) de {@link SubModuloUsaTrayectoria#setTrayectoriaModificada(Trayectoria) setTrayectoriaModificada(Trayectoria)} 
 * y de la función de resultado
 * que le solicita el motor.</strong>
 * <br>Ha de ser avisdo cuando el motor actua o para a través de {@link ModificadorTrayectoria#actuar() actuar()}
 * y {@link ModificadorTrayectoria#parar() parar()} respectivamente.
 * </li>
 * </ul>
 * <br>
 * Si alguno de los submódulos necesita la trayectoria para sus cálculos deberá pedriselo al motor a través de 
 * {@link #apuntaNecesitaTrayectoria(SubModuloUsaTrayectoria)}, con ello el motor sabra que hay módulos que la necesitan
 * y los apuntará en una lista. 
 * Cada vez que comienze a {@link #actuar()} el motor deberá pedir la trayectoria inicial a {@link VentanasMonitoriza},
 * que puede haber cambiado desde al ultima actuación, y se la indica a los submódulos 
 * con {@link SubModuloUsaTrayectoria#setTrayectoriaInicial(Trayectoria) setTrayectoriaInicial(Trayectoria)} .
 * <br>
 * Los submódulos que necesitan una trayectoria serán avisados,
 * invocando {@link SubModuloUsaTrayectoria#setTrayectoriaModificada(Trayectoria) setTrayectoriaModificada(Trayectoria)},
 * si la trayectoria se modifica con respecto a la última fijada. Esto ocurrirá, asíncronamente cuando el 
 *  {@link ModificadorTrayectoria} tenga una nueva trayectoria (como ya se mencionó). 
 * <br>
 * También el motor debe tener un modelo de coche ({@link Coche}) que actualizá con los datos recibidos de los sensores.
 * <br>
 * Finalmente con la información de consigna de dirección y velocidad, y obstáculos detectados, recibida de los submódulos,
 * y según el algoritmo implementado, mandará las consigna calculada al coche real. 
 * 
 * 
 * @author alberto
 *
 */
public interface Motor extends Modulo {
	
	
	/** Cuando se invoca este método el método el motor debe empezar a actuar sobre el carro */
	public void actuar();
	
	/** Cunando este método es invocado el motor debe dejar de actuar sobre el carro */
	public void parar();

	/** Establece el {@link CalculoVelocidad} a utilizar */
	public void setCalculadorVelocidad(CalculoVelocidad calVel);
	
	/** Establece el {@link CalculoVelocidad} a utilizar */
	public void setCalculadorDireccion(CalculoDireccion calDir);
	
	/** Establece los {@link DetectaObstaculos} a utilizar */
	public void setDetectaObstaculos(DetectaObstaculos[] dectObs);
	
	/** Establece el {@link ModificadorTrayectoria} a utilizar (si se va a utilizar alguno)*/
	public void setModificadorTrayectoria(ModificadorTrayectoria modifTr);
	
	/** El motor debe llevar un modelo el coche que actualiza con cada nuevo dato de GPS, IMU, etc.
	 * Los modulos consultan el estado del coche a través de este modelo
	 */
	public Coche getModeloCoche();
	
	/** Los {@link SubModuloUsaTrayectoria} se apuntan en el motor para recibir la trayectoria inicial al 
	 * comenzar a actuar. Con ello el motor sabrá 
	 * que hay alguno que necesita trayectorias y se las pedirá a {@link VentanasMonitoriza} 
	 */
	public void apuntaNecesitaTrayectoria(SubModuloUsaTrayectoria smutr);
		
	/** Será invocado por el {@link ModificadorTrayectoria} cada vez que tenga una nueva trayectoria.
	 * El {@link Motor} deberá limitarse a invocar el método equivalente de todos a quellos {@link SubModuloUsaTrayectoria}
	 * que se hayan apuntado a través de {@link #apuntaNecesitaTrayectoria(SubModuloUsaTrayectoria)}.
	 */
	public void nuevaTrayectoria(Trayectoria nuevaTr);
}
