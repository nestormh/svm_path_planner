/**
 * 
 */
package sibtra.ui.modulos;

import sibtra.ui.VentanasMonitoriza;
import sibtra.util.LabelDato;
import sibtra.util.LabelDatoFormato;
import sibtra.util.ManejaJoystick;
import sibtra.util.PanelDatos;
import sibtra.util.PanelFlow;
import sibtra.util.PanelJoystick;
import sibtra.util.ThreadSupendible;

/**
 * Módulo que usa joystick para tomar velocidad y dirección
 * @author alberto
 *
 */
public class NavegaJoystick implements CalculoDireccion, CalculoVelocidad {
	
	private final static String NOMBRE="Navega Joystick";
	private final static String DESCRIPCION="Usa el Joystick para definir direccion y velocidad";
	protected static final long milisActulizacion = 200;
	
	private VentanasMonitoriza ventMonitoriza;
	private ManejaJoystick manJoy;
	private ThreadSupendible thCiclico;
	private PanelJoystick panJoy;

	public NavegaJoystick(VentanasMonitoriza ventMoni) {
		ventMonitoriza=ventMoni;
		manJoy=new ManejaJoystick();
		panJoy=new PanelJoystick(manJoy);

		ventMonitoriza.añadePanel(panJoy, "Joystick", false);
		
		thCiclico=new ThreadSupendible() {
			@Override
			protected void accion() {
				panJoy.actualiza(); //esto ya realiza el pool
				try { Thread.sleep(milisActulizacion); } catch (InterruptedException e) {}
			}
		};
		thCiclico.setName(NOMBRE);
		
	}

	/** @return directamente la velocidad que me da {@link ManejaJoystick} */
	public double getConsignaVelocidad() {
		return manJoy.getVelocidad();
	}

	/** @return directamente el alfa que calcula {@link ManejaJoystick} */
	public double getConsignaDireccion() {
		return manJoy.getAlfa();
	}

	/**
	 * @see sibtra.ui.modulos.Modulo#getDescripcion()
	 */
	public String getDescripcion() {
		return DESCRIPCION;
	}

	/**
	 * @see sibtra.ui.modulos.Modulo#getNombre()
	 */
	public String getNombre() {
		return NOMBRE;
	}

	/** Suspendemos el {@link #thCiclico} */
	public void terminar() {
		thCiclico.suspender();
	}

}
